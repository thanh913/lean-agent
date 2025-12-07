"""Public entrypoints for the rebuilt MiniF2F decomposition environment."""

from __future__ import annotations

import hashlib
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.types import State

from .env import LeanDecomposeEnv


# ---------------------------------------------------------------------------
# Planner System Prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are the Planner in a Lean 4 proving system. You work alongside a Prover: you design high-level proof blueprints, and the Prover fills in low-level Lean tactics. Your job is to enforce proof structure (cases, helper lemmas, reductions) while delegating the technical details.

## Blueprint Requirements
Attempt **incrementally** at designing a high-level Lean 4 proof (blueprint) for the Prover to fill up the low-level details.
Requirements:
1. If first attempt: **Always** use a single `defer` for the entire goal.
2. If failure: From environment's feedback, diagnose the root cause from the error, then introduce *just enough* structure to fix it.
3. Repeat until the proof succeeds.

To submit a blueprint, you must use the `<blueprint>...</blueprint>` tag. Inside that tag, include a single fenced `lean4` code block with your current blueprint. Example:

<blueprint>
```lean4
theorem NAME (binders) : GOAL := by
  -- your use of `defer` goes here
  defer
```
</blueprint>

## The `defer` Interface
Syntax: `defer h₁ h₂ ... hₙ`

Hands the current goal in scope goal to the Prover, who sees only the goal plus the listed hypotheses (and their type dependencies). Pass only what is logically required.
```lean
-- First attempt: defer the whole goal.
theorem div6_consecutive (n : ℤ) : 6 ∣ n * (n + 1) * (n + 2) := by
  defer

-- Pass hypotheses the goal depends on (dependencies like `b` are included automatically).
theorem dvd_trans_example (a b c : ℤ) (hab : a ∣ b) (hbc : b ∣ c) : a ∣ c := by
  defer hab hbc

-- Decomposition after a failed first attempt:
theorem sum_two_squares_ne_3 (x y : ℤ) : x^2 + y^2 ≠ 3 := by
  by_contra h
  have hx : x^2 % 4 = 0 ∨ x^2 % 4 = 1 := by defer
  have hy : y^2 % 4 = 0 ∨ y^2 % 4 = 1 := by defer
  have hsum : (x^2 + y^2) % 4 = 3 := by defer h
  defer hx hy hsum
```

Natural structures to use: `have`, `suffices`, `obtain`, `constructor`, `cases`, `induction`, and `calc`. Limit yourself to at most eight subgoals per plan.

## Type Discipline
- Numeric literals default to `ℕ`. Cast the first literal when working over `ℤ`, `ℚ`, or `ℝ`.
- `(5 : ℕ) - 7 = 0`; cast to `ℤ` first when real subtraction is needed.
- `(7 : ℤ) / 2 = 3`; use `ℚ` or `ℝ` when you need exact division.
- Once a variable has type `ℝ`, `x + 1` and similar expressions pick up the right literal type automatically.

## Answer Format:
First, write from first principles a proof of the theorem, so as to refer to in later turns. After that, iterate turn-by-turn on the blueprint following the "Blueprint Requirements".
"""


def _split_header_and_snippet(code: str) -> tuple[list[str], str]:
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


def _final_verification_reward(*, state: State, **_: Any) -> float:
    return 1.0 if state.get("final_verification") else 0.0


def _blueprint_compiled_reward(*, state: State, **_: Any) -> float:
    attempts = state.get("blueprint_compile_attempts", 0) or 0
    successes = state.get("blueprint_compile_successes", 0) or 0
    if not attempts:
        return 0.0
    return float(successes) / float(attempts)


def _prover_calls_reward(*, state: State, **_: Any) -> float:
    return float(state.get("prover_calls", 0) or 0)


def _planner_tokens_reward(*, state: State, **_: Any) -> float:
    prompt = state.get("planner_prompt_tokens", 0) or 0
    completion = state.get("planner_completion_tokens", 0) or 0
    return float(prompt + completion)


def _prover_tokens_reward(*, state: State, **_: Any) -> float:
    prompt = state.get("prover_prompt_tokens", 0) or 0
    completion = state.get("prover_completion_tokens", 0) or 0
    return float(prompt + completion)


def _planner_turns_reward(*, state: State, **_: Any) -> float:
    return float(state.get("planner_turns", 0) or 0)


def _planner_time_reward(*, state: State, **_: Any) -> float:
    # Stored internally in milliseconds; expose as seconds with 1 decimal.
    ms = state.get("planner_inference_ms", 0.0) or 0.0
    sec = float(ms) / 1000.0
    return float(round(sec, 1))


def _prover_time_reward(*, state: State, **_: Any) -> float:
    # Stored internally in milliseconds; expose as seconds with 1 decimal.
    ms = state.get("prover_inference_ms", 0.0) or 0.0
    sec = float(ms) / 1000.0
    return float(round(sec, 1))


def load_environment(
    *,
    verification_url: str = "",
    verification_api_key_env: str = "VERIFICATION_KEY",
    planner_budget: int = 1,
    verify_timeout: int = 60,
    max_prover_attempts: int = 2,
    prover_model: str = "",
    prover_base_url: str | None = None,
    max_parallel_prover: int = 8,
    dataset_name: str = "AI-MO/minif2f_test",
    split: str | None = "train",
    limit: int | None = None,
    prover_api_key_env: str = "OPENAI_API_KEY",
    prover_sampling: dict[str, Any] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    if not prover_model:
        raise ValueError("prover_model must be provided to load_environment")
    if max_parallel_prover < 1:
        raise ValueError("max_parallel_prover must be at least 1")

    split_name = split or "train"
    hf_dataset = load_dataset(dataset_name, split=split_name)
    # Optional: randomize example order (used with vf-eval -n for random subsets).
    shuffle = bool(kwargs.pop("shuffle", False))
    shuffle_seed = kwargs.pop("shuffle_seed", None)
    if shuffle:
        if shuffle_seed is not None:
            try:
                hf_dataset = hf_dataset.shuffle(seed=int(shuffle_seed))
            except Exception:
                hf_dataset = hf_dataset.shuffle()
        else:
            hf_dataset = hf_dataset.shuffle()
    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(hf_dataset):
        header_lines, theorem_snippet = _split_header_and_snippet(item.get("formal_statement", ""))
        if not theorem_snippet:
            continue
        question = f"```lean4\n{theorem_snippet}\n```"
        planner_prompt = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        sample_name = str(item.get("name") or "").strip()
        session_key = sample_name or hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
        rows.append(
            {
                "question": question,
                "prompt": planner_prompt,
                "answer": "",
                "info": {
                    "header_lines": header_lines,
                    "verification_url": verification_url,
                    "session_id": f"minif2f-{session_key}-{idx}",
                },
            }
        )

    dataset = Dataset.from_list(rows)

    rubric = vf.Rubric(
        funcs=[
            _final_verification_reward,
            _blueprint_compiled_reward,
            _prover_calls_reward,
            _planner_turns_reward,
            _planner_tokens_reward,
            _prover_tokens_reward,
            _planner_time_reward,
            _prover_time_reward,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    env_kwargs = {
        "verification_url": verification_url,
        "verification_api_key_env": verification_api_key_env,
        "planner_budget": planner_budget,
        "verify_timeout": verify_timeout,
        "max_prover_attempts": max_prover_attempts,
        "prover_model": prover_model,
        "prover_base_url": prover_base_url,
        "prover_api_key_env": prover_api_key_env,
        "max_parallel_prover": max_parallel_prover,
        "prover_sampling": prover_sampling,
    }

    return LeanDecomposeEnv(
        env_id="minif2f_decompose",
        dataset=dataset,
        rubric=rubric,
        **env_kwargs,
        **kwargs,
    )


__all__ = ["LeanDecomposeEnv", "load_environment"]
