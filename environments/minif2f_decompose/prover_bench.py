#!/usr/bin/env python3
"""Prover benchmark: directly attempt MiniF2F problems with the prover model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI

from minif2f_decompose.clients import LeanClient, LLMClient, create_httpx_client
from minif2f_decompose.executor import (
    PROVER_PROMPT_TEMPLATE,
    extract_last_lean_block,
    extract_proof_body,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """A benchmark problem."""

    name: str
    header_lines: list[str]
    theorem_snippet: str
    formal_statement: str


@dataclass
class Attempt:
    """Result of a single proof attempt."""

    success: bool
    proof_body: str | None
    error: str | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ProblemResult:
    """Result for a single problem."""

    name: str
    success: bool
    attempts: list[Attempt]
    total_latency_ms: float
    final_proof: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""

    total: int
    solved: int
    failed: int
    total_attempts: int
    total_latency_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    problems: list[ProblemResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.solved / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Benchmark Results",
            f"-----------------",
            f"Total problems: {self.total}",
            f"Solved: {self.solved} ({self.pass_rate:.1%})",
            f"Failed: {self.failed}",
            f"Total attempts: {self.total_attempts}",
            f"Total time: {self.total_latency_s:.1f}s",
            f"Total prompt tokens: {self.total_prompt_tokens:,}",
            f"Total completion tokens: {self.total_completion_tokens:,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Problem Loading
# ---------------------------------------------------------------------------


def split_header_and_snippet(code: str) -> tuple[list[str], str]:
    """Split formal statement into header lines and theorem snippet."""
    lines = [ln.rstrip("\n") for ln in str(code).splitlines()]
    idx = 0
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith(("theorem ", "lemma ", "def ", "example ", "corollary ")):
            idx = i
            break
    header_lines = [ln for ln in lines[:idx]]
    theorem_snippet = "\n".join(lines[idx:]).strip()
    return header_lines, theorem_snippet


def load_problems(
    dataset_name: str = "AI-MO/minif2f_test",
    split: str = "train",
    limit: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[Problem]:
    """Load problems from HuggingFace dataset."""
    hf_dataset = load_dataset(dataset_name, split=split)

    if shuffle:
        if seed is not None:
            hf_dataset = hf_dataset.shuffle(seed=seed)
        else:
            hf_dataset = hf_dataset.shuffle()

    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    problems: list[Problem] = []
    for item in hf_dataset:
        formal_statement = item.get("formal_statement", "")
        header_lines, theorem_snippet = split_header_and_snippet(formal_statement)
        if not theorem_snippet:
            continue
        name = str(item.get("name") or f"problem_{len(problems)}")
        problems.append(
            Problem(
                name=name,
                header_lines=header_lines,
                theorem_snippet=theorem_snippet,
                formal_statement=formal_statement,
            )
        )

    return problems


# ---------------------------------------------------------------------------
# Proof Verification
# ---------------------------------------------------------------------------


def build_verify_code(problem: Problem, proof_body: str) -> str:
    """Build complete Lean code for verification."""
    # Start with imports
    imports = []
    seen = set()
    for line in problem.header_lines:
        stripped = line.strip()
        if stripped.startswith("import ") and stripped not in seen:
            imports.append(stripped)
            seen.add(stripped)
    if "import Mathlib" not in seen:
        imports.insert(0, "import Mathlib")

    # Get the theorem header (everything up to and including := by)
    snippet_lines = problem.theorem_snippet.splitlines()
    header_lines = []
    for i, line in enumerate(snippet_lines):
        header_lines.append(line)
        if ":= by" in line or line.strip().endswith(":= by"):
            break

    # Build the full code
    code_lines = imports + [""] + header_lines

    # Add the proof body with proper indentation
    for line in proof_body.strip().splitlines():
        if line.strip():
            code_lines.append(f"  {line}")
        else:
            code_lines.append("")

    return "\n".join(code_lines) + "\n"


def format_snippet_with_sorry(problem: Problem) -> str:
    """Format the theorem snippet with sorry for the prover prompt."""
    lines = problem.theorem_snippet.splitlines()
    result = []
    found_by = False

    for line in lines:
        result.append(line)
        if ":= by" in line or line.strip().endswith(":= by"):
            found_by = True
            break

    if found_by:
        result.append("  sorry")

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------


class ProverBench:
    """Benchmark runner for the prover."""

    def __init__(
        self,
        llm: LLMClient,
        lean: LeanClient,
        *,
        model: str,
        sampling: dict[str, Any],
        max_attempts: int = 1,
        verify_timeout: int = 120,
        stagger_delay: float = 60.0,
    ) -> None:
        self._llm = llm
        self._lean = lean
        self._model = model
        self._sampling = sampling
        self._max_attempts = max_attempts
        self._verify_timeout = verify_timeout
        self._stagger_delay = stagger_delay

    async def solve_problem(self, problem: Problem) -> ProblemResult:
        """Attempt to solve a single problem with staggered attempts.

        Only spawns the next attempt after the previous one has actually
        started executing (acquired the semaphore) AND stagger_delay has
        elapsed since that start. This prevents queueing all attempts at
        the semaphore under congestion.
        """
        snippet = format_snippet_with_sorry(problem)
        prompt = PROVER_PROMPT_TEMPLATE.format(snippet=snippet.strip())
        stagger_delay = self._stagger_delay
        loop = asyncio.get_running_loop()

        start_time = time.time()
        pending: set[asyncio.Task] = set()
        started_events: list[asyncio.Event] = []
        attempts: list[Attempt] = []
        next_attempt_idx = 0
        watched_attempt_idx = -1  # Which attempt we're waiting to start
        start_time_of_watched: float | None = None  # When watched attempt started

        def _collect_done(done_tasks: set[asyncio.Task]) -> ProblemResult | None:
            """Process completed tasks, return ProblemResult if success found."""
            nonlocal attempts
            for task in done_tasks:
                try:
                    attempt_result = task.result()
                except Exception as e:
                    attempts.append(
                        Attempt(
                            success=False,
                            proof_body=None,
                            error=str(e),
                            latency_ms=0,
                            prompt_tokens=0,
                            completion_tokens=0,
                        )
                    )
                    continue

                attempts.append(attempt_result["attempt"])

                if attempt_result["success"]:
                    return ProblemResult(
                        name=problem.name,
                        success=True,
                        attempts=attempts,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        final_proof=attempt_result["verify_code"],
                    )
            return None

        def _spawn_attempt() -> None:
            """Spawn the next attempt with a started_event."""
            nonlocal next_attempt_idx, watched_attempt_idx, start_time_of_watched
            event = asyncio.Event()
            started_events.append(event)
            task = asyncio.create_task(
                self._single_attempt(problem, next_attempt_idx, prompt, started_event=event),
                name=f"{problem.name}-attempt{next_attempt_idx}",
            )
            pending.add(task)
            watched_attempt_idx = next_attempt_idx
            start_time_of_watched = None  # Will be set when event fires
            next_attempt_idx += 1

        def _check_watched_started() -> bool:
            """Check if watched attempt has started, update timing if so."""
            nonlocal start_time_of_watched
            if watched_attempt_idx < 0 or watched_attempt_idx >= len(started_events):
                return False
            event = started_events[watched_attempt_idx]
            if event.is_set() and start_time_of_watched is None:
                start_time_of_watched = loop.time()
                return True
            return start_time_of_watched is not None

        def _can_spawn_next() -> bool:
            """Check if we can spawn the next attempt."""
            if next_attempt_idx >= self._max_attempts:
                return False
            if next_attempt_idx == 0:
                return True  # Always spawn first immediately
            # Need watched attempt to have started + stagger_delay elapsed
            if start_time_of_watched is None:
                return False
            elapsed = loop.time() - start_time_of_watched
            return elapsed >= stagger_delay

        # Main loop
        while next_attempt_idx < self._max_attempts or pending:
            # Spawn next attempt if conditions are met
            if _can_spawn_next():
                _spawn_attempt()

            if not pending:
                break

            # Determine timeout for wait
            timeout: float | None = None
            if next_attempt_idx < self._max_attempts:
                if start_time_of_watched is not None:
                    # Waiting for stagger_delay to elapse
                    elapsed = loop.time() - start_time_of_watched
                    timeout = max(0.01, stagger_delay - elapsed)
                else:
                    # Waiting for watched attempt to start - poll periodically
                    timeout = 0.1

            # Wait for task completion or timeout
            try:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                break

            # Check if watched attempt has started
            _check_watched_started()

            # Process completed tasks
            if done:
                result = _collect_done(done)
                if result:
                    # Cancel remaining pending attempts
                    for p in pending:
                        p.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    return result

        # All attempts exhausted or failed
        total_latency = (time.time() - start_time) * 1000
        return ProblemResult(
            name=problem.name,
            success=False,
            attempts=attempts,
            total_latency_ms=total_latency,
        )

    async def _single_attempt(
        self,
        problem: Problem,
        attempt_idx: int,
        prompt: str,
        started_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        """Execute a single proof attempt. Returns dict with result info.

        Args:
            problem: The problem being solved
            attempt_idx: Attempt number (0-indexed)
            prompt: The prover prompt
            started_event: Optional event to set when LLM call actually starts
                          (after acquiring semaphore)
        """
        attempt = Attempt(
            success=False,
            proof_body=None,
            error=None,
            latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
        )

        # Call LLM
        try:
            response = await self._llm.call(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                request_id=f"{problem.name}-attempt{attempt_idx}",
                started_event=started_event,
                **self._sampling,
            )
        except Exception as e:
            attempt = Attempt(
                success=False,
                proof_body=None,
                error=f"LLM error: {e}",
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
            )
            return {"success": False, "attempt": attempt, "verify_code": None}

        attempt = Attempt(
            success=False,
            proof_body=None,
            error=None,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

        # Extract proof body
        lean_block = extract_last_lean_block(response.content)
        if not lean_block:
            attempt = Attempt(
                success=False,
                proof_body=None,
                error="No Lean code block in response",
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            return {"success": False, "attempt": attempt, "verify_code": None}

        proof_body = extract_proof_body(lean_block)
        if not proof_body.strip():
            attempt = Attempt(
                success=False,
                proof_body=None,
                error="No proof body extracted",
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            return {"success": False, "attempt": attempt, "verify_code": None}

        # Verify the proof
        verify_code = build_verify_code(problem, proof_body)
        verify_result = await self._lean.compile(
            code=verify_code,
            timeout=self._verify_timeout,
            allow_sorry=False,
            snippet_id=f"{problem.name}-verify{attempt_idx}",
        )

        if verify_result.ok:
            attempt = Attempt(
                success=True,
                proof_body=proof_body,
                error=None,
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            return {"success": True, "attempt": attempt, "verify_code": verify_code}
        else:
            attempt = Attempt(
                success=False,
                proof_body=proof_body,
                error=verify_result.error_log or "Verification failed",
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            return {"success": False, "attempt": attempt, "verify_code": None}

    async def run_benchmark(
        self,
        problems: list[Problem],
        *,
        max_parallel: int = 8,
        progress_callback: Any = None,
    ) -> BenchmarkResult:
        """Run the benchmark on all problems."""
        sem = asyncio.Semaphore(max_parallel)

        async def solve_with_sem(problem: Problem) -> ProblemResult:
            async with sem:
                result = await self.solve_problem(problem)
                if progress_callback:
                    progress_callback(result)
                return result

        results = await asyncio.gather(
            *[solve_with_sem(p) for p in problems],
            return_exceptions=True,
        )

        # Process results
        problem_results: list[ProblemResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                problem_results.append(
                    ProblemResult(
                        name=problems[i].name,
                        success=False,
                        attempts=[],
                        total_latency_ms=0,
                    )
                )
            else:
                problem_results.append(result)

        # Aggregate
        solved = sum(1 for r in problem_results if r.success)
        total_attempts = sum(len(r.attempts) for r in problem_results)
        total_latency = sum(r.total_latency_ms for r in problem_results)
        total_prompt = sum(
            sum(a.prompt_tokens for a in r.attempts) for r in problem_results
        )
        total_completion = sum(
            sum(a.completion_tokens for a in r.attempts) for r in problem_results
        )

        return BenchmarkResult(
            total=len(problems),
            solved=solved,
            failed=len(problems) - solved,
            total_attempts=total_attempts,
            total_latency_s=total_latency / 1000,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            problems=problem_results,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prover benchmark for MiniF2F problems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        default="AI-MO/minif2f_test",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit number of problems",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle problems",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling",
    )

    # Model options
    parser.add_argument(
        "--model",
        default=None,
        help="Prover model (default: from PROVER_MODEL env)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Prover API base URL (default: from PROVER_BASE_URL env)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        required=True,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Max tokens to generate",
    )

    # Execution options
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Max proof attempts per problem",
    )
    parser.add_argument(
        "--stagger-delay",
        type=float,
        default=60.0,
        help="Seconds to wait before starting next attempt (staggered execution)",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=8,
        help="Max concurrent problems",
    )
    parser.add_argument(
        "--max-parallel-llm",
        type=int,
        default=64,
        help="Global cap on concurrent LLM calls",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=120,
        help="Lean verification timeout (seconds)",
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Environment file
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file",
    )

    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    # Load environment (shell script already sources .env, this is a fallback)
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()  # Tries .env in cwd

    # Get configuration from environment
    prover_model = args.model or os.getenv("PROVER_MODEL", "")
    prover_base_url = args.base_url or os.getenv("PROVER_BASE_URL", "")
    prover_key = os.getenv("PROVER_KEY", "")
    verification_url = os.getenv("VERIFICATION_URL", "")
    verification_key = os.getenv("VERIFICATION_KEY", "")

    if not prover_model:
        print("Error: PROVER_MODEL not set", file=sys.stderr)
        return 1
    if not verification_url:
        print("Error: VERIFICATION_URL not set", file=sys.stderr)
        return 1

    print(f"Prover model: {prover_model}")
    print(f"Prover URL: {prover_base_url or 'default'}")
    print(f"Verification URL: {verification_url}")
    print()

    # Load problems
    print(f"Loading problems from {args.dataset}...")
    problems = load_problems(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Loaded {len(problems)} problems")
    print()

    # Create clients (same setup as decompose env)
    http_client = create_httpx_client()
    try:
        openai_client = AsyncOpenAI(
            api_key=prover_key,
            base_url=prover_base_url or None,
            http_client=http_client,
        )
        llm = LLMClient(
            openai_client,
            max_parallel=args.max_parallel_llm,
            max_retries=0,  # Retry handled at task level
            track_concurrency=True,  # Debug: track prover concurrency
        )
        lean = LeanClient(
            verification_url,
            verification_key,
            max_retries=2,
        )

        sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }

        # Create benchmark runner
        bench = ProverBench(
            llm=llm,
            lean=lean,
            model=prover_model,
            sampling=sampling,
            max_attempts=args.max_attempts,
            verify_timeout=args.verify_timeout,
            stagger_delay=args.stagger_delay,
        )

        # Progress tracking
        solved_count = 0
        total_count = 0

        def on_progress(result: ProblemResult) -> None:
            nonlocal solved_count, total_count
            total_count += 1
            if result.success:
                solved_count += 1
            status = "PASS" if result.success else "FAIL"
            attempts_used = len(result.attempts)
            print(f"[{total_count}/{len(problems)}] {result.name}: {status} ({attempts_used}/{args.max_attempts} attempts)")
            if args.verbose and not result.success and result.attempts:
                last_attempt = result.attempts[-1]
                if last_attempt.error:
                    print(f"  Error: {last_attempt.error[:200]}")

        # Run benchmark
        print("Running benchmark...")
        print("-" * 40)
        start_time = time.time()

        result = await bench.run_benchmark(
            problems,
            max_parallel=args.concurrency,
            progress_callback=on_progress,
        )

        elapsed = time.time() - start_time
        print("-" * 40)
        print()
        print(result.summary())
        print(f"Wall time: {elapsed:.1f}s")

        # Print attempt distribution for solved problems
        solved_attempts = [len(p.attempts) for p in result.problems if p.success]
        if solved_attempts:
            avg_attempts = sum(solved_attempts) / len(solved_attempts)
            print(f"\nAttempts per solved problem: avg={avg_attempts:.2f}")
            # Distribution: how many solved at each attempt count
            dist = Counter(solved_attempts)
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
            print(f"Distribution (attempts:count): {dist_str}")

        # Collect all attempt latencies for distribution analysis
        all_latencies_ms = [
            a.latency_ms
            for p in result.problems
            for a in p.attempts
            if a.latency_ms > 0  # Only include completed attempts
        ]

        # Print latency distribution
        latency_stats = {}
        if all_latencies_ms:
            sorted_latencies = sorted(all_latencies_ms)
            n = len(sorted_latencies)

            def percentile(p: float) -> float:
                idx = int(p / 100 * (n - 1))
                return sorted_latencies[idx]

            latency_stats = {
                "count": n,
                "min_ms": min(sorted_latencies),
                "max_ms": max(sorted_latencies),
                "mean_ms": statistics.mean(sorted_latencies),
                "median_ms": statistics.median(sorted_latencies),
                "p50_ms": percentile(50),
                "p90_ms": percentile(90),
                "p95_ms": percentile(95),
                "p99_ms": percentile(99),
            }

            print(f"\nInference Latency Distribution ({n} attempts):")
            print(f"  Min:    {latency_stats['min_ms']/1000:6.1f}s")
            print(f"  P50:    {latency_stats['p50_ms']/1000:6.1f}s")
            print(f"  P90:    {latency_stats['p90_ms']/1000:6.1f}s")
            print(f"  P95:    {latency_stats['p95_ms']/1000:6.1f}s")
            print(f"  P99:    {latency_stats['p99_ms']/1000:6.1f}s")
            print(f"  Max:    {latency_stats['max_ms']/1000:6.1f}s")
            print(f"  Mean:   {latency_stats['mean_ms']/1000:6.1f}s")

        # Save results
        if args.output:
            output_data = {
                "config": {
                    "dataset": args.dataset,
                    "split": args.split,
                    "limit": args.limit,
                    "model": prover_model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "max_attempts": args.max_attempts,
                    "stagger_delay_s": args.stagger_delay,
                },
                "summary": {
                    "total": result.total,
                    "solved": result.solved,
                    "failed": result.failed,
                    "pass_rate": result.pass_rate,
                    "total_attempts": result.total_attempts,
                    "total_latency_s": result.total_latency_s,
                    "wall_time_s": elapsed,
                },
                "latency_distribution": latency_stats,
                "attempt_latencies_ms": all_latencies_ms,  # Raw data for analysis
                "problems": [
                    {
                        "name": p.name,
                        "success": p.success,
                        "attempts": len(p.attempts),
                        "latency_ms": p.total_latency_ms,
                        "attempt_latencies_ms": [a.latency_ms for a in p.attempts],
                    }
                    for p in result.problems
                ],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")

        return 0 if result.solved > 0 else 1
    finally:
        await http_client.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
