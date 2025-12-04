"""Public entrypoints for the MiniF2F prover-only benchmark environment."""

from __future__ import annotations

from typing import Any

import verifiers as vf
from verifiers.types import State
from datasets import Dataset, load_dataset

from minif2f_decompose.system_prompt import PLANNER_SYSTEM_PROMPT as _PLANNER_PROMPT  # for reference only
from .env import ProverBenchEnv


def _split_header_and_snippet(code: str) -> tuple[list[str], str]:
    lines = [ln.rstrip("\n") for ln in str(code).splitlines()]
    idx = 0
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(("theorem ", "lemma ", "def ", "example ", "corollary ")):
            idx = i
            break
    header_lines = [ln for ln in lines[:idx]]
    theorem_snippet = "\n".join(lines[idx:]).strip()
    return header_lines, theorem_snippet


def _final_verification_reward(*, state: State, **_: Any) -> float:
    return 1.0 if state.get("final_verification") else 0.0


def _prover_calls_reward(*, state: State, **_: Any) -> float:
    return float(state.get("prover_calls", 0) or 0)


def load_environment(
    *,
    verification_url: str = "",
    verification_api_key_env: str = "VERIFICATION_KEY",
    verify_timeout: int = 60,
    max_prover_attempts: int = 2,
    prover_model: str = "",
    prover_base_url: str | None = None,
    prover_api_key_env: str = "OPENAI_API_KEY",
    max_parallel_prover: int = 8,
    dataset_name: str = "AI-MO/minif2f_test",
    split: str | None = "train",
    limit: int | None = None,
    prover_sampling: dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    if not prover_model:
        raise ValueError("prover_model must be provided to load_environment")
    if max_parallel_prover < 1:
        raise ValueError("max_parallel_prover must be at least 1")

    split_name = split or "train"
    hf = load_dataset(dataset_name, split=split_name)
    if limit is not None:
        hf = hf.select(range(min(limit, len(hf))))
    rows: list[dict[str, Any]] = []
    for item in hf:
        header_lines, theorem_snippet = _split_header_and_snippet(item.get("formal_statement", ""))
        q = f"```lean4\n{theorem_snippet}\n```"
        # Keep the planner prompt as context in question for traceability, but env ignores it.
        planner_prompt = [
            {"role": "system", "content": _PLANNER_PROMPT},
            {"role": "user", "content": q},
        ]
        rows.append(
            {
                "question": q,
                "prompt": planner_prompt,
                "answer": "",
                "info": {
                    "header_lines": header_lines,
                    "theorem_snippet": theorem_snippet,
                    "verification_url": verification_url,
                },
            }
        )

    dataset = Dataset.from_list(rows)

    rubric = vf.Rubric(
        funcs=[_final_verification_reward, _prover_calls_reward],
        weights=[1.0, 0.0],
    )

    env_args = {
        "verification_url": verification_url,
        "verification_api_key_env": verification_api_key_env,
        "verify_timeout": verify_timeout,
        "max_prover_attempts": max_prover_attempts,
        "prover_model": prover_model,
        "prover_base_url": prover_base_url,
        "max_parallel_prover": max_parallel_prover,
        "prover_api_key_env": prover_api_key_env,
        "prover_sampling": prover_sampling,
    }

    return ProverBenchEnv(
        env_id="minif2f_prover_bench",
        dataset=dataset,
        rubric=rubric,
        **env_args,
        **kwargs,
    )


__all__ = ["ProverBenchEnv", "load_environment"]
