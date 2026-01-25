"""Public entrypoints for the lean_agent environment."""

from __future__ import annotations

from typing import Any

import verifiers as vf
from datasets import Dataset

from .env import LeanAgentEnv
from .prompts import SYSTEM_PROMPT, format_user_prompt
from .test_dataset import TEST_DATASET


# Reward functions
def _verified_reward(*, state: Any, **_: Any) -> float:
    """Reward: 1.0 if proof was verified, 0.0 otherwise."""
    return 1.0 if state.get("verified") else 0.0


def _turns_reward(*, state: Any, **_: Any) -> float:
    """Metric: number of turns taken."""
    return float(len(state.get("trajectory", [])))


def _subagent_calls_reward(*, state: Any, **_: Any) -> float:
    """Metric: number of subagent calls made."""
    return float(state.get("subagent_calls", 0))


def load_environment(
    *,
    verification_url: str = "",
    verification_api_key_env: str = "VERIFICATION_KEY",
    verify_timeout: int = 60,
    max_turns: int = 10,
    subagent_max_turns: int = 5,
    dataset_name: str | None = None,
    split: str | None = "train",
    limit: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load the lean_agent environment.

    Args:
        verification_url: URL for the Lean verification service.
        verification_api_key_env: Environment variable name for verification API key.
        verify_timeout: Timeout in seconds for Lean compilation.
        max_turns: Maximum turns for the main model.
        subagent_max_turns: Maximum turns per subagent.
        dataset_name: HuggingFace dataset name, or None to use TEST_DATASET.
        split: Dataset split to use.
        limit: Limit number of examples.
        shuffle: Whether to shuffle the dataset.
        shuffle_seed: Seed for shuffling.
        **kwargs: Additional arguments passed to the environment.

    Returns:
        Configured LeanAgentEnv instance.
    """
    # Load dataset
    if dataset_name:
        from datasets import load_dataset

        split_name = split or "train"
        hf_dataset = load_dataset(dataset_name, split=split_name)

        if shuffle:
            if shuffle_seed is not None:
                hf_dataset = hf_dataset.shuffle(seed=int(shuffle_seed))
            else:
                hf_dataset = hf_dataset.shuffle()

        if limit is not None:
            hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

        # Convert to our format
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(hf_dataset):
            theorem = item.get("theorem", "")
            nl_proof = item.get("nl_proof", "")
            header_lines = item.get("header_lines", ["import Mathlib"])

            if not theorem:
                continue

            # Build user prompt
            question = format_user_prompt(theorem, nl_proof, header_lines)

            rows.append({
                "question": question,
                "answer": "",
                "info": {"verification_url": verification_url},
            })
    else:
        # Use test dataset
        rows = []
        for idx, item in enumerate(TEST_DATASET):
            theorem = item["theorem"]
            nl_proof = item["nl_proof"]
            header_lines = item["header_lines"]

            question = format_user_prompt(theorem, nl_proof, header_lines)

            rows.append({
                "question": question,
                "answer": "",
                "info": {"verification_url": verification_url},
            })

        if limit is not None:
            rows = rows[:limit]

    dataset = Dataset.from_list(rows)

    # Build rubric
    rubric = vf.Rubric(
        funcs=[
            _verified_reward,
            _turns_reward,
            _subagent_calls_reward,
        ],
        weights=[1.0, 0.0, 0.0],
    )

    # Default sampling args with stop sequences
    # Note: Don't include max_completion_tokens - it breaks some APIs like Gemini
    default_sampling = {
        "stop": ["</submit>", "</prove>"],
    }

    # Merge with any provided sampling args
    sampling_args = kwargs.pop("sampling_args", None) or {}
    merged_sampling = {**default_sampling, **sampling_args}

    return LeanAgentEnv(
        env_id="lean_agent",
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=max_turns,
        verification_url=verification_url,
        verification_api_key_env=verification_api_key_env,
        verify_timeout=verify_timeout,
        subagent_max_turns=subagent_max_turns,
        sampling_args=merged_sampling,
        **kwargs,
    )


__all__ = ["LeanAgentEnv", "load_environment"]
