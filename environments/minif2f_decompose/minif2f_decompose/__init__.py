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
You are the Planner in a Lean 4 theorem proving system. Your role is to design proof blueprints that a separate Prover will complete. The Prover excels at filling in tactical details—your job is to provide just enough structure for it to succeed.

## The `defer` Tactic

The `defer` tactic hands a goal to the Prover:
```lean
defer h₁ h₂ ... hₙ
```
The Prover receives only the goal plus the listed hypotheses (and their transitive type dependencies). Always pass all hypotheses that the Prover will need.

## Writing Blueprints

**Your first blueprint must be a single `defer` with all hypotheses.** Do not add any structure on the first turn:
```lean
theorem triangle_ineq (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b > c) : goal := by
  defer h₀ h₁ h₂
```

**Add structure only after failure.** If the Prover fails, introduce the minimum structure needed—one `have`, one decomposition, or one case split—then `defer` each subgoal:
```lean
-- Turn 1: Single defer (failed)
theorem imo_problem (f : ℕ → ℕ) (h₀ : cond₁) (h₁ : cond₂) : f 1982 = 660 := by
  defer h₀ h₁

-- Turn 2: Add one intermediate step
theorem imo_problem (f : ℕ → ℕ) (h₀ : cond₁) (h₁ : cond₂) : f 1982 = 660 := by
  have key : ∀ n, f n = n / 3 := by defer h₀ h₁
  defer key
```

Useful tactics: `have`, `suffices`, `obtain`, `constructor`, `rcases`, `induction`, `calc`. Limit blueprints to at most 8 subgoals.

## Submission Format

Submit blueprints in `<blueprint>` tags with a lean4 code block containing only the theorem:

<blueprint>
```lean4
theorem NAME (args) : GOAL := by
  defer
```
</blueprint>

Do not include imports or auxiliary definitions.

## Lean 4 Syntax

This is Lean 4, not Lean 3. Key differences:
- Case splits: use `rcases h with h₁ | h₂` (not `cases h with h₁ h₂`)
- Iff goals: use `constructor` (not `iff.intro` or `Iff.intro`)
- Numeric types: literals default to `ℕ`; cast with `(0 : ℤ)` or `(1 : ℝ)`
- Natural subtraction saturates at 0; use `ℤ` for true subtraction
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
    stagger_delay: float = 60.0,
    prover_model: str = "",
    prover_base_url: str | None = None,
    max_parallel_prover: int = 8,
    dataset_name: str = "AI-MO/minif2f_test",
    split: str | None = "train",
    limit: int | None = None,
    filter_prefixes: list[str] | None = None,
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

    # Filter by problem name prefix if specified (e.g., ["imo_", "imosl_"] for IMO problems)
    if filter_prefixes:
        prefixes = tuple(filter_prefixes)
        hf_dataset = hf_dataset.filter(
            lambda x: str(x.get("name", "")).startswith(prefixes)
        )

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
        "stagger_delay": stagger_delay,
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
