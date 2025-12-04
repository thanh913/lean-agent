"""Environment scaffold for the rebuilt MiniF2F decomposition task."""

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

from .proof_executor import ExecutionResult, execute_plan
from .prover import OpenAIProver


async def _planner_turn(
    *,
    client: AsyncOpenAI,
    model: str,
    transcript: list[dict[str, str]],
    sampling_args: dict[str, Any] | None = None,
) -> tuple[str, int, int, float]:
    """Call the planner model once, appending its reply to the transcript.

    Returns (content, prompt_tokens, completion_tokens, latency_ms) based on the provider's
    usage accounting, defaulting to zeros if unavailable.
    """
    payload = {k: v for k, v in (sampling_args or {}).items() if v is not None}
    loop = asyncio.get_running_loop()
    start = loop.time()
    response = await client.chat.completions.create(
        model=model,
        messages=transcript,
        **payload,
    )
    elapsed_ms = (loop.time() - start) * 1000.0
    content = response.choices[0].message.content or ""
    # If the model stopped at the `</blueprint>` stop sequence without echoing
    # the terminator, synthesize it so downstream XML parsing always sees a
    # complete <blueprint>...</blueprint> region.
    if "<blueprint>" in content and "</blueprint>" not in content:
        content = content + "</blueprint>"
    transcript.append({"role": "assistant", "content": content})
    usage = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return content, prompt_tokens, completion_tokens, elapsed_ms


def _record_step(state: State, prompt_part: Messages, completion_part: Messages, kind: str) -> None:
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


def _format_executor_feedback(result: ExecutionResult) -> str:
    prefix_map = {
        "success": "Plan compiled: executor finished successfully.",
        "plan_compile": "Plan failed to compile.",
        "task_failure": "Plan compiled but some subgoals failed.",
        "final_verify": "Plan solved subgoals but final verification failed.",
    }
    lines: list[str] = [prefix_map.get(result.stage, "Executor feedback")]  # type: ignore[arg-type]
    if result.subgoals:
        lines[0] += f" ({result.subgoals} subgoal(s))"
    if result.detail:
        lines.append(result.detail.strip())
    body = "\n".join(line for line in lines if line)
    proof_block = ""
    if result.proof_block:
        proof_block = f"\n<Proof>\n{result.proof_block.strip()}\n</Proof>"
    return f"<REPL>{body}{proof_block}</REPL>"


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


@dataclass
class RunConfig:
    header_lines: list[str]
    backend_url: str
    planner_budget: int
    verify_timeout: int
    max_prover_attempts: int
    max_parallel_prover: int
    prover_model: str
    prover_base_url: str | None
    prover_sampling: dict[str, Any]
    session_id: str


@dataclass
class RunMetrics:
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

    def record_planner_turn(self, prompt_tokens: int, completion_tokens: int, latency_ms: float) -> None:
        self.planner_turns += 1
        self.planner_prompt_tokens += prompt_tokens
        self.planner_completion_tokens += completion_tokens
        self.planner_inference_ms += latency_ms

    def record_executor_result(self, result: ExecutionResult) -> None:
        self.prover_calls += result.attempts
        self.prover_prompt_tokens += result.prover_prompt_tokens
        self.prover_completion_tokens += result.prover_completion_tokens
        self.prover_inference_ms += result.prover_inference_ms
        self.blueprint_attempts += 1
        compiled_ok = result.stage != "plan_compile"
        if compiled_ok:
            self.blueprint_successes += 1
        self.last_blueprint_compiled = compiled_ok
        if result.success:
            self.final_verification = True

    def write_to_state(self, state: State) -> None:
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


class LeanDecomposeEnv(Environment):
    """Planner/executor environment for the MiniF2F defer-based task."""

    def __init__(
        self,
        *,
        backend_url: str = "",
        planner_budget: int = 1,
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
        self.backend_url = backend_url
        self.planner_budget = planner_budget
        self.verify_timeout = verify_timeout
        self.max_prover_attempts = max_prover_attempts
        self.prover_model = prover_model
        self.prover_base_url = prover_base_url
        self.prover_api_key_env = prover_api_key_env
        self.max_parallel_prover = max_parallel_prover
        self.prover_sampling = prover_sampling or {}

    def _build_run_config(self, info: dict[str, Any]) -> RunConfig:
        header_lines = list(info.get("header_lines") or [])
        backend_url = str(info.get("backend_url", self.backend_url) or "").strip()
        planner_budget = int(info.get("planner_budget", self.planner_budget) or 1)
        verify_timeout = int(info.get("verify_timeout", self.verify_timeout))
        max_prover_attempts = int(info.get("max_prover_attempts", self.max_prover_attempts))
        max_parallel_prover = int(info.get("max_parallel_prover", self.max_parallel_prover) or 1)
        prover_sampling = info.get("prover_sampling", self.prover_sampling) or {}
        prover_model = str(info.get("prover_model", self.prover_model)).strip()
        base_override = info.get("prover_base_url", self.prover_base_url)
        prover_base_url = str(base_override or "").strip() or None
        session_id = str(info.get("session_id") or "plan")
        return RunConfig(
            header_lines=header_lines,
            backend_url=backend_url,
            planner_budget=max(1, planner_budget),
            verify_timeout=verify_timeout,
            max_prover_attempts=max_prover_attempts,
            max_parallel_prover=max(1, max_parallel_prover),
            prover_model=prover_model,
            prover_base_url=prover_base_url,
            prover_sampling=prover_sampling,
            session_id=session_id,
        )

    def _build_prover(self, config: RunConfig) -> OpenAIProver:
        prover_key = os.getenv(self.prover_api_key_env, "")
        prover_client = AsyncOpenAI(api_key=prover_key, base_url=config.prover_base_url)
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

    async def setup_state(self, state: State) -> State:  # pragma: no cover - hook for subclasses
        return state

    async def rollout(
        self,
        input,
        client,
        model: str,
        sampling_args: dict[str, Any] | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        prompt: Messages = state["prompt"]
        if not isinstance(prompt, list):
            raise ValueError("prompt must be a list of chat messages for this env")

        info: dict[str, Any] = dict(state.get("info") or {})
        config = self._build_run_config(info)
        prover = self._build_prover(config)
        metrics = RunMetrics()

        planner_messages: list[dict[str, str]] = list(prompt)
        original_prompt_len = len(planner_messages)

        for _ in range(config.planner_budget):
            before_call = list(planner_messages)
            plan_text, planner_ptoks, planner_ctoks, planner_ms = await _planner_turn(
                client=client,
                model=model,
                transcript=planner_messages,
                sampling_args=sampling_args,
            )
            metrics.record_planner_turn(planner_ptoks, planner_ctoks, planner_ms)
            if len(planner_messages) > len(before_call):
                _record_step(state, before_call, [planner_messages[-1]], "planner")

            state["plan_text"] = plan_text
            blueprint_text = _extract_blueprint(plan_text)
            if not blueprint_text:
                rejection = (
                    "<REPL>Plan rejected: your reply must contain a "
                    "<blueprint>...</blueprint> tag with Lean code inside.</REPL>"
                )
                feedback_msg = {"role": "user", "content": rejection}
                history_before = list(planner_messages)
                planner_messages.append(feedback_msg)
                _record_step(state, history_before, [feedback_msg], "plan_invalid")
                state["execution_summary"] = rejection
                continue

            result = await execute_plan(
                plan_text=blueprint_text,
                header_lines=config.header_lines,
                prover=prover,
                backend_url=config.backend_url,
                verify_timeout=config.verify_timeout,
                max_prover_attempts=config.max_prover_attempts,
                session_id=config.session_id,
            )
            metrics.record_executor_result(result)

            feedback = _format_executor_feedback(result)
            feedback_msg = {"role": "user", "content": feedback}
            history_before = list(planner_messages)
            planner_messages.append(feedback_msg)
            _record_step(state, history_before, [feedback_msg], "executor_feedback")
            state["execution_summary"] = feedback

            if result.success:
                break

        state["completion"] = planner_messages[original_prompt_len:]
        metrics.write_to_state(state)

        elapsed_ms = (time.time() - state["timing"]["start_time"]) * 1000
        state["timing"]["generation_ms"] = elapsed_ms
        state["timing"]["total_ms"] = elapsed_ms
        state["is_completed"] = True
        state["stop_condition"] = "finished"
        return state
