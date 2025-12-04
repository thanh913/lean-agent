"""Async prover abstraction plus an OpenAI-backed implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from openai import AsyncOpenAI

from .lean_utils import extract_last_lean_block, extract_proof_body


@dataclass
class ProverRequest:
    theorem_snippet: str
    session_id: str


@dataclass
class ProverResult:
    success: bool
    body: str | None
    attempts: int
    log: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inference_ms: float = 0.0


class Prover(Protocol):
    async def prove(self, request: ProverRequest, max_attempts: int = 1) -> ProverResult: ...


class OpenAIProver:
    """Simple wrapper that prompts an OpenAI chat model for Lean proofs."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        sampling: dict[str, Any] | None = None,
        max_parallel: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        default_sampling = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_completion_tokens": 20000,
        }
        if sampling:
            default_sampling.update({k: v for k, v in sampling.items() if v is not None})
        self.sampling = default_sampling
        self._sem = asyncio.Semaphore(max(1, max_parallel))

    async def prove(self, request: ProverRequest, max_attempts: int = 1) -> ProverResult:
        prompt = self._build_prompt(request.theorem_snippet)
        messages = [{"role": "user", "content": prompt}]
        attempts = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_ms = 0.0
        async with self._sem:
            for _ in range(max(1, max_attempts)):
                attempts += 1
                try:
                    start = asyncio.get_running_loop().time()
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **self.sampling,
                    )
                    total_ms += (asyncio.get_running_loop().time() - start) * 1000.0
                    usage = getattr(response, "usage", None)
                    pt = ct = 0
                    if usage is not None:
                        if isinstance(usage, dict):
                            pt = int(usage.get("prompt_tokens") or 0)
                            ct = int(usage.get("completion_tokens") or 0)
                        else:
                            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                            ct = int(getattr(usage, "completion_tokens", 0) or 0)
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                    content = response.choices[0].message.content or ""
                except Exception as exc:  # pragma: no cover - transport errors
                    return ProverResult(
                        False,
                        None,
                        attempts,
                        f"Prover error: {exc}",
                        total_prompt_tokens,
                        total_completion_tokens,
                        total_ms,
                    )

                snippet = extract_last_lean_block(content)
                if not snippet:
                    continue
                body = extract_proof_body(snippet)
                if body.strip():
                    return ProverResult(
                        True,
                        body.strip(),
                        attempts,
                        "",
                        total_prompt_tokens,
                        total_completion_tokens,
                        total_ms,
                    )
        return ProverResult(
            False,
            None,
            attempts,
            "No Lean proof body produced",
            total_prompt_tokens,
            total_completion_tokens,
            total_ms,
        )

    def _build_prompt(self, snippet: str) -> str:
        return (
            "Complete the following Lean 4 code:\n\n"
            "```lean4\n"
            f"{snippet.strip()}```\n\n"
            "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan "
            "outlining the main proof steps and strategies.\n"
            "The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the "
            "construction of the final formal proof."
        )
