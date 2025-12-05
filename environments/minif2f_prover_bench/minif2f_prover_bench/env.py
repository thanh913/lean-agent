"""Prover-only benchmark environment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
import os
import time

from verifiers.envs.environment import Environment
from verifiers.types import Messages, State, TrajectoryStep
from openai import AsyncOpenAI

from minif2f_decompose.http_utils import create_httpx_client
from minif2f_decompose.lean_utils import inject_proof_body, lean_compile
from minif2f_decompose.prover import OpenAIProver, ProverRequest, ProverResult
from .lean_utils import parse_theorem_signature


def _join_header_and_snippet(header_lines: list[str], snippet: str) -> str:
    """Merge optional header lines with a theorem snippet, trimming outer whitespace."""
    if not header_lines:
        return snippet.strip()
    combined = header_lines + ["", snippet]
    return "\n".join(combined).strip()


@dataclass
class BenchConfig:
    header_lines: list[str]
    theorem_snippet: str
    verification_url: str
    verification_key: str
    verify_timeout: int
    max_prover_attempts: int
    max_parallel_prover: int
    prover_sampling: dict[str, Any]
    prover_model: str
    prover_base_url: str | None
    session_id: str


class ProverBenchEnv(Environment):
    """Run the prover directly on MiniF2F snippets (no planner)."""

    def __init__(
        self,
        *,
        verification_url: str = "",
        verification_api_key_env: str = "VERIFICATION_KEY",
        verify_timeout: int = 60,
        max_prover_attempts: int = 2,
        prover_model: str = "",
        prover_base_url: str | None = None,
        prover_api_key_env: str = "OPENAI_API_KEY",
        max_parallel_prover: int = 8,
        prover_sampling: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.verification_url = verification_url
        self.verification_api_key_env = verification_api_key_env
        self.verify_timeout = verify_timeout
        self.max_prover_attempts = max_prover_attempts
        self.prover_model = prover_model
        self.prover_base_url = prover_base_url
        self.prover_api_key_env = prover_api_key_env
        self.max_parallel_prover = max_parallel_prover
        self.prover_sampling = prover_sampling or {}

    def _build_config(self, info: dict[str, Any]) -> BenchConfig:
        header_lines = list(info.get("header_lines") or [])
        theorem_snippet = str(info.get("theorem_snippet", "")).strip()
        verification_url = str(info.get("verification_url", self.verification_url) or "").strip()
        verification_key = os.getenv(self.verification_api_key_env, "")
        verify_timeout = int(info.get("verify_timeout", self.verify_timeout))
        max_prover_attempts = int(info.get("max_prover_attempts", self.max_prover_attempts))
        max_parallel_prover = int(info.get("max_parallel_prover", self.max_parallel_prover) or 1)
        prover_sampling = info.get("prover_sampling", self.prover_sampling) or {}
        prover_model = str(info.get("prover_model", self.prover_model)).strip()
        base_override = info.get("prover_base_url", self.prover_base_url)
        prover_base_url = str(base_override or "").strip() or None
        session_id = str(info.get("session_id") or "prover-bench")
        return BenchConfig(
            header_lines=header_lines,
            theorem_snippet=theorem_snippet,
            verification_url=verification_url,
            verification_key=verification_key,
            verify_timeout=verify_timeout,
            max_prover_attempts=max(1, max_prover_attempts),
            max_parallel_prover=max(1, max_parallel_prover),
            prover_sampling=prover_sampling,
            prover_model=prover_model,
            prover_base_url=prover_base_url,
            session_id=session_id,
        )

    def _build_prover(self, config: BenchConfig) -> OpenAIProver:
        prover_key = os.getenv(self.prover_api_key_env, "")
        http_client = create_httpx_client()
        prover_client = AsyncOpenAI(
            api_key=prover_key,
            base_url=config.prover_base_url,
            http_client=http_client,
        )
        return OpenAIProver(
            client=prover_client,
            model=config.prover_model,
            sampling=config.prover_sampling,
            max_parallel=config.max_parallel_prover,
        )

    async def init_state(
        self,
        input,
        client,
        model: str,
        sampling_args: dict[str, Any] | None = None,
    ) -> State:
        state = await super().init_state(input, client, model, sampling_args)
        state.update(
            {
                "final_verification": False,
                "prover_calls": 0,
                "prover_failures": 0,
                "execution_summary": "",
            }
        )
        return state

    async def setup_state(self, state: State) -> State:
        return state

    async def rollout(
        self,
        input,
        client,
        model: str,
        sampling_args: dict[str, Any] | None = None,
    ) -> State:  # type: ignore[override]
        state = await self.init_state(input, client, model, sampling_args)
        prompt: Messages = state["prompt"]
        if not isinstance(prompt, list):
            raise ValueError("prompt must be a list of chat messages for this env")

        info: dict[str, Any] = dict(state.get("info") or {})
        config = self._build_config(info)
        prover = self._build_prover(config)

        success, detail, attempts = await self._prove_and_verify(config=config, prover=prover)

        state["prover_calls"] = attempts
        state["prover_failures"] = 0 if success else 1
        state["final_verification"] = bool(success)

        feedback = f"<REPL>{detail}</REPL>"
        feedback_msg = {"role": "user", "content": feedback}

        step: TrajectoryStep = {
            "prompt": list(prompt),
            "completion": [feedback_msg],
            "response": None,
            "tokens": None,
            "reward": None,
            "advantage": None,
            "extras": {"kind": "prover"},
        }
        state["trajectory"].append(step)
        state["completion"] = [feedback_msg]
        state["execution_summary"] = feedback

        elapsed_ms = (time.time() - state["timing"]["start_time"]) * 1000
        state["timing"]["generation_ms"] = elapsed_ms
        state["timing"]["total_ms"] = elapsed_ms
        state["is_completed"] = True
        state["stop_condition"] = "finished"
        return state

    async def _prove_and_verify(
        self,
        *,
        config: BenchConfig,
        prover: OpenAIProver,
    ) -> tuple[bool, str, int]:
        """Run prover attempts with streaming parallelism and early termination.

        Maintains max_parallel_prover in-flight at all times. As each prover call
        completes, immediately verify if successful. On first verified success,
        cancel all remaining tasks and return.
        """
        full_snippet = _join_header_and_snippet(config.header_lines, config.theorem_snippet)
        attempts = 0
        last_log = ""

        try:
            parse_theorem_signature(config.theorem_snippet)
        except Exception as exc:
            return False, f"Invalid theorem snippet: {exc}", attempts

        # Track state
        success_found = False
        success_detail = ""
        pending_tasks: set[asyncio.Task] = set()
        next_attempt_idx = 0

        async def prove_and_verify_one(attempt_idx: int) -> tuple[bool, str, int]:
            """Run one prover attempt and verify if successful. Returns (ok, log, attempts)."""
            req = ProverRequest(
                theorem_snippet=full_snippet,
                session_id=f"{config.session_id}-{attempt_idx}",
            )
            result = await prover.prove(req, max_attempts=1)
            log = result.log or ""

            body = (result.body or "").strip()
            if not result.success or not body:
                return False, log, result.attempts

            # Got a proof body, verify it
            candidate = inject_proof_body(config.theorem_snippet, body)
            final_code = _join_header_and_snippet(config.header_lines, candidate) + "\n"
            compile_result = await lean_compile(
                code=final_code,
                verification_url=config.verification_url,
                verification_key=config.verification_key,
                timeout=config.verify_timeout,
                allow_sorry=False,
                snippet_id=f"{config.session_id}-attempt{attempt_idx}",
            )

            if compile_result.ok:
                return True, f"Prover succeeded at attempt {attempt_idx + 1}.", result.attempts
            return False, compile_result.log or log, result.attempts

        def spawn_next() -> asyncio.Task | None:
            """Spawn the next prover task if attempts remain."""
            nonlocal next_attempt_idx
            if next_attempt_idx >= config.max_prover_attempts:
                return None
            idx = next_attempt_idx
            next_attempt_idx += 1
            task = asyncio.create_task(prove_and_verify_one(idx))
            pending_tasks.add(task)
            return task

        # Initial burst: fill up to max_parallel_prover
        for _ in range(min(config.max_parallel_prover, config.max_prover_attempts)):
            spawn_next()

        # Process as they complete
        while pending_tasks and not success_found:
            done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                pending_tasks.discard(task)
                try:
                    ok, log, task_attempts = task.result()
                    attempts += task_attempts
                    last_log = log or last_log

                    if ok:
                        success_found = True
                        success_detail = log
                        break
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    last_log = f"Task error: {exc}"

            # If not done, spawn replacement to keep pipeline full
            if not success_found:
                spawn_next()

        # Cancel any remaining tasks
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if success_found:
            return True, success_detail, attempts

        detail = "Prover failed."
        if last_log:
            detail = f"{detail}\n{last_log}"
        return False, detail, attempts
