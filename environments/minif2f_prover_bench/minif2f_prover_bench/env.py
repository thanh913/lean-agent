"""Prover-only benchmark environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os
import time

from verifiers.envs.environment import Environment
from verifiers.types import Messages, State, TrajectoryStep
from openai import AsyncOpenAI

from minif2f_decompose.lean_utils import inject_proof_body, lean_compile
from minif2f_decompose.prover import OpenAIProver, ProverRequest
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
    backend_url: str
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
        backend_url: str = "",
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
        backend_url = str(info.get("backend_url", self.backend_url) or "").strip()
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
            backend_url=backend_url,
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
        full_snippet = _join_header_and_snippet(config.header_lines, config.theorem_snippet)
        attempts = 0
        last_log = ""

        try:
            parse_theorem_signature(config.theorem_snippet)
        except Exception as exc:
            return False, f"Invalid theorem snippet: {exc}", attempts

        for attempt in range(config.max_prover_attempts):
            req = ProverRequest(
                theorem_snippet=full_snippet,
                session_id=f"{config.session_id}-{attempt}",
            )
            result = await prover.prove(req, max_attempts=1)
            attempts += result.attempts
            last_log = result.log
            body = (result.body or "").strip()
            if not result.success or not body:
                continue
            candidate = inject_proof_body(config.theorem_snippet, body)
            final_code = _join_header_and_snippet(config.header_lines, candidate) + "\n"
            compile_result = await lean_compile(
                code=final_code,
                backend_url=config.backend_url,
                timeout=config.verify_timeout,
                allow_sorry=False,
                snippet_id=f"{config.session_id}-attempt{attempt}",
            )
            if compile_result.ok:
                detail = f"Prover succeeded in {attempts} attempt(s)."
                return True, detail, attempts
            last_log = compile_result.log

        detail = "Prover failed."
        if last_log:
            detail = f"{detail}\n{last_log}"
        return False, detail, attempts
