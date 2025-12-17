"""Environment scaffold for the MiniF2F decomposition task."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
import verifiers as vf

from verifiers.envs.environment import Environment
from verifiers.types import Messages, State, TrajectoryStep

from .clients import LeanClient, LLMClient, create_httpx_client
from .executor import ExecutionResult, ExecutorConfig, ProofExecutor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Per-run configuration for the planner/executor loop."""

    header_lines: list[str]
    verification_url: str
    verification_key: str
    planner_budget: int
    verify_timeout: int
    max_prover_attempts: int
    max_parallel_prover: int
    stagger_delay: float
    prover_model: str
    prover_base_url: str | None
    prover_sampling: dict[str, Any]
    session_id: str


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """Accumulated metrics for a run."""

    planner_turns: int = 0
    planner_prompt_tokens: int = 0
    planner_completion_tokens: int = 0
    planner_inference_ms: float = 0.0
    prover_prompt_tokens: int = 0
    prover_completion_tokens: int = 0
    prover_inference_ms: float = 0.0
    prover_calls: int = 0
    blueprint_attempts: int = 0
    blueprint_successes: int = 0
    last_blueprint_compiled: bool = False
    final_verification: bool = False

    def record_planner(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record a planner turn."""
        self.planner_turns += 1
        self.planner_prompt_tokens += prompt_tokens
        self.planner_completion_tokens += completion_tokens
        self.planner_inference_ms += latency_ms

    def record_execution(self, result: ExecutionResult) -> None:
        """Record executor result."""
        self.prover_calls += result.attempts
        self.prover_prompt_tokens += result.prover_prompt_tokens
        self.prover_completion_tokens += result.prover_completion_tokens
        self.prover_inference_ms += result.prover_inference_ms
        self.blueprint_attempts += 1
        compiled_ok = result.stage != "compile"
        if compiled_ok:
            self.blueprint_successes += 1
        self.last_blueprint_compiled = compiled_ok
        if result.success:
            self.final_verification = True

    def write_to_state(self, state: State) -> None:
        """Write metrics to state dict."""
        state["planner_turns"] = self.planner_turns
        state["planner_prompt_tokens"] = self.planner_prompt_tokens
        state["planner_completion_tokens"] = self.planner_completion_tokens
        state["planner_inference_ms"] = self.planner_inference_ms
        state["prover_prompt_tokens"] = self.prover_prompt_tokens
        state["prover_completion_tokens"] = self.prover_completion_tokens
        state["prover_inference_ms"] = self.prover_inference_ms
        state["prover_calls"] = self.prover_calls
        state["blueprint_compile_attempts"] = self.blueprint_attempts
        state["blueprint_compile_successes"] = self.blueprint_successes
        state["blueprint_compiled"] = self.last_blueprint_compiled
        state["final_verification"] = self.final_verification


# ---------------------------------------------------------------------------
# Blueprint Extraction
# ---------------------------------------------------------------------------


_BLUEPRINT_PARSER = vf.XMLParser(fields=["blueprint"], answer_field="blueprint")


def _extract_blueprint(plan_text: str) -> str | None:
    """Extract the content of the <blueprint> tag, if present."""
    if not plan_text:
        return None
    parsed = _BLUEPRINT_PARSER.parse(plan_text, last=True)
    blueprint = getattr(parsed, "blueprint", None)
    if blueprint is None:
        return None
    stripped = blueprint.strip()
    return stripped or None


def _fix_blueprint_tag(content: str) -> str:
    """Synthesize closing tag if model stopped at stop sequence."""
    if "<blueprint>" in content and "</blueprint>" not in content:
        return content + "</blueprint>"
    return content


# ---------------------------------------------------------------------------
# Feedback Formatting
# ---------------------------------------------------------------------------


def _format_executor_feedback(result: ExecutionResult) -> str:
    """Format executor result as feedback for the planner."""
    prefix_map = {
        "done": "Plan compiled: executor finished successfully.",
        "compile": "Plan failed to compile.",
        "solve": "Plan compiled but some subgoals failed.",
        "verify": "Plan solved subgoals but final verification failed.",
    }
    lines: list[str] = [prefix_map.get(result.stage, "Executor feedback")]
    if result.subgoals:
        lines[0] += f" ({result.subgoals} subgoal(s))"
    if result.detail:
        lines.append(result.detail.strip())
    body = "\n".join(line for line in lines if line)
    proof_block = ""
    if result.proof_block:
        proof_block = f"\n<Proof>\n{result.proof_block.strip()}\n</Proof>"
    return f"<REPL>{body}{proof_block}</REPL>"


def _rejection_message() -> dict[str, str]:
    """Build rejection message for invalid blueprint."""
    return {
        "role": "user",
        "content": (
            "<REPL>Plan rejected: your reply must contain a "
            "<blueprint>...</blueprint> tag with Lean code inside.</REPL>"
        ),
    }


# ---------------------------------------------------------------------------
# Trajectory Recording
# ---------------------------------------------------------------------------


def _record_step(
    state: State,
    prompt_part: Messages,
    completion_part: Messages,
    kind: str,
) -> None:
    """Record a trajectory step."""
    step: TrajectoryStep = {
        "prompt": prompt_part,
        "completion": completion_part,
        "response": None,
        "tokens": None,
        "reward": None,
        "advantage": None,
        "extras": {"kind": kind},
    }
    state["trajectory"].append(step)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LeanDecomposeEnv(Environment):
    """Planner/executor environment for the MiniF2F defer-based task."""

    def __init__(
        self,
        *,
        verification_url: str = "",
        verification_api_key_env: str = "VERIFICATION_KEY",
        planner_budget: int = 1,
        verify_timeout: int = 60,
        max_prover_attempts: int = 2,
        stagger_delay: float = 60.0,
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
        self.planner_budget = planner_budget
        self.verify_timeout = verify_timeout
        self.max_prover_attempts = max_prover_attempts
        self.stagger_delay = stagger_delay
        self.prover_model = prover_model
        self.prover_base_url = prover_base_url
        self.prover_api_key_env = prover_api_key_env
        self.max_parallel_prover = max_parallel_prover
        self.prover_sampling = prover_sampling or {}
        # Shared semaphore for prover concurrency across all rollouts
        # Created eagerly to avoid any race conditions
        self._prover_semaphore = asyncio.Semaphore(max(1, max_parallel_prover))
        # Track how many env instances are created (should be exactly 1!)
        if not hasattr(LeanDecomposeEnv, '_instance_count'):
            LeanDecomposeEnv._instance_count = 0
        LeanDecomposeEnv._instance_count += 1
        print(f"[ENV #{LeanDecomposeEnv._instance_count}] Created prover semaphore with max_parallel={max_parallel_prover}")

    def _build_run_config(self, info: dict[str, Any]) -> RunConfig:
        """Build run configuration from instance defaults and per-example info."""
        return RunConfig(
            header_lines=list(info.get("header_lines") or []),
            verification_url=str(
                info.get("verification_url", self.verification_url) or ""
            ).strip(),
            verification_key=os.getenv(self.verification_api_key_env, ""),
            planner_budget=max(
                1, int(info.get("planner_budget", self.planner_budget) or 1)
            ),
            verify_timeout=int(info.get("verify_timeout", self.verify_timeout)),
            max_prover_attempts=int(
                info.get("max_prover_attempts", self.max_prover_attempts)
            ),
            max_parallel_prover=max(
                1, int(info.get("max_parallel_prover", self.max_parallel_prover) or 1)
            ),
            stagger_delay=float(info.get("stagger_delay", self.stagger_delay)),
            prover_model=str(info.get("prover_model", self.prover_model)).strip(),
            prover_base_url=str(
                info.get("prover_base_url", self.prover_base_url) or ""
            ).strip()
            or None,
            prover_sampling=info.get("prover_sampling", self.prover_sampling) or {},
            session_id=str(info.get("session_id") or "plan"),
        )

    def _get_prover_semaphore(self) -> asyncio.Semaphore:
        """Get the shared prover semaphore (created in __init__)."""
        return self._prover_semaphore

    def _build_prover_llm(self, config: RunConfig) -> tuple[LLMClient, "httpx.AsyncClient"]:
        """Build LLM client for the prover.

        Returns:
            Tuple of (LLMClient, httpx.AsyncClient). Caller must close the httpx client.
        """
        import httpx  # for type hint

        prover_key = os.getenv(self.prover_api_key_env, "")
        http_client = create_httpx_client()
        openai_client = AsyncOpenAI(
            api_key=prover_key,
            base_url=config.prover_base_url,
            http_client=http_client,
        )
        # Use shared semaphore across all rollouts for global concurrency control
        shared_sem = self._get_prover_semaphore()
        llm_client = LLMClient(
            openai_client,
            max_retries=0,  # Retry handled at task level
            semaphore=shared_sem,
            track_concurrency=True,  # Debug: track prover concurrency
        )
        return llm_client, http_client

    def _build_lean_client(self, config: RunConfig) -> LeanClient:
        """Build Lean compilation client."""
        return LeanClient(
            config.verification_url,
            config.verification_key,
            max_retries=2,
        )

    async def setup_state(self, state: State) -> State:
        """Initialize environment-specific state fields."""
        state.update(
            {
                "plan_text": None,
                "execution_summary": "",
                "final_verification": False,
                "planner_turns": 0,
                "blueprint_compiled": False,
                "blueprint_compile_attempts": 0,
                "blueprint_compile_successes": 0,
                "planner_prompt_tokens": 0,
                "planner_completion_tokens": 0,
                "planner_inference_ms": 0.0,
                "prover_prompt_tokens": 0,
                "prover_completion_tokens": 0,
                "prover_inference_ms": 0.0,
                "prover_calls": 0,
            }
        )
        return state

    async def rollout(
        self,
        input,
        client,
        model: str,
        sampling_args: dict[str, Any] | None = None,
    ) -> State:
        """Run the planner/executor loop."""
        state = await self.init_state(input, client, model, sampling_args)
        state = await self.setup_state(state)
        prompt: Messages = state["prompt"]
        if not isinstance(prompt, list):
            raise ValueError("prompt must be a list of chat messages for this env")

        info: dict[str, Any] = dict(state.get("info") or {})
        config = self._build_run_config(info)

        # Build clients
        planner_llm = LLMClient(client, max_parallel=1)  # Planner is sequential
        prover_llm, prover_http_client = self._build_prover_llm(config)
        lean = self._build_lean_client(config)

        try:
            executor = ProofExecutor(
                prover_llm,
                lean,
                ExecutorConfig(
                    prover_model=config.prover_model,
                    prover_sampling=config.prover_sampling,
                    verify_timeout=config.verify_timeout,
                    max_prover_attempts=config.max_prover_attempts,
                    session_id=config.session_id,
                    stagger_delay=config.stagger_delay,
                ),
            )

            metrics = Metrics()
            transcript: list[dict[str, str]] = list(prompt)
            original_prompt_len = len(transcript)

            # Main planner loop
            for turn in range(config.planner_budget):
                before_call = list(transcript)

                # 1. Call planner
                response = await planner_llm.call(
                    model=model,
                    messages=transcript,
                    request_id=f"{config.session_id}-planner-{turn}",
                    **(sampling_args or {}),
                )
                content = _fix_blueprint_tag(response.content)
                transcript.append({"role": "assistant", "content": content})
                metrics.record_planner(
                    response.prompt_tokens,
                    response.completion_tokens,
                    response.latency_ms,
                )

                if len(transcript) > len(before_call):
                    _record_step(state, before_call, [transcript[-1]], "planner")

                state["plan_text"] = content

                # 2. Extract blueprint
                blueprint = _extract_blueprint(content)
                if not blueprint:
                    rejection = _rejection_message()
                    history_before = list(transcript)
                    transcript.append(rejection)
                    _record_step(state, history_before, [rejection], "plan_invalid")
                    state["execution_summary"] = rejection["content"]
                    continue

                # 3. Execute blueprint
                result = await executor.execute(blueprint, config.header_lines)
                metrics.record_execution(result)

                # 4. Format and append feedback
                feedback_content = _format_executor_feedback(result)
                feedback_msg = {"role": "user", "content": feedback_content}
                history_before = list(transcript)
                transcript.append(feedback_msg)
                _record_step(state, history_before, [feedback_msg], "executor_feedback")
                state["execution_summary"] = feedback_content

                if result.success:
                    break

            # Finalize state
            state["completion"] = transcript[original_prompt_len:]
            metrics.write_to_state(state)

            elapsed_ms = (time.time() - state["timing"]["start_time"]) * 1000
            state["timing"]["generation_ms"] = elapsed_ms
            state["timing"]["total_ms"] = elapsed_ms
            state["is_completed"] = True
            state["stop_condition"] = "finished"

            return state
        finally:
            await prover_http_client.aclose()
