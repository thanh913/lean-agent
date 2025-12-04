from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pass_at_prover_calls import _ensure_graph_dir, _label_from_meta, _load_metadata


def _load_results(output_dir: Path) -> pd.DataFrame:
    results_path = output_dir / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"results.jsonl not found in {output_dir}")
    rows: List[Dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {results_path}")
    return pd.DataFrame(rows)


def _compute_pass_curve(metric: np.ndarray, success: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pass@B for a non-negative cost metric (tokens or time)."""
    n = len(success)
    if n == 0:
        raise ValueError("Empty dataset")
    # Unique budgets in ascending order
    budgets = np.unique(metric)
    budgets = budgets[~np.isnan(budgets)]
    budgets = np.sort(budgets)
    if budgets.size == 0:
        return np.array([0.0]), np.array([float(success.mean())])
    pass_rates: List[float] = []
    for b in budgets:
        solved_within_b = (success == 1.0) & (metric <= b)
        pass_b = float(solved_within_b.sum()) / float(n)
        pass_rates.append(pass_b)
    return budgets, np.array(pass_rates, dtype=float)


def _load_df_and_meta(output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = _load_results(output_dir)
    meta = _load_metadata(output_dir)
    return df, meta


def generate_for_run(output_dir: Path) -> None:
    """Generate token/time vs accuracy curves for a single vf-eval run.

    Only runs that expose the decompose token/time metrics will produce graphs.
    """
    df, meta = _load_df_and_meta(output_dir)
    required_token_cols = {"_planner_tokens_reward", "_prover_tokens_reward", "_final_verification_reward"}
    required_time_cols = {"_planner_time_reward", "_prover_time_reward", "_final_verification_reward"}

    # If metrics are missing (e.g., prover-only env or older runs), skip gracefully.
    has_tokens = required_token_cols.issubset(df.columns)
    has_time = required_time_cols.issubset(df.columns)
    if not has_tokens and not has_time:
        return

    success = df["_final_verification_reward"].fillna(0).to_numpy(dtype=float)
    graph_dir = _ensure_graph_dir(output_dir)
    label = _label_from_meta(meta)

    if has_tokens:
        planner_tokens = df["_planner_tokens_reward"].fillna(0).to_numpy(dtype=float)
        prover_tokens = df["_prover_tokens_reward"].fillna(0).to_numpy(dtype=float)
        total_tokens = planner_tokens + prover_tokens
        xs_tok, ys_tok = _compute_pass_curve(total_tokens, success)
        xs_pl, ys_pl = _compute_pass_curve(planner_tokens, success)

        plt.figure(figsize=(5, 4))
        plt.plot(xs_tok, ys_tok, linestyle="-")
        plt.xlabel("Total tokens budget (planner + prover)")
        plt.ylabel("Pass@tokens")
        plt.ylim(0.5, 1.0)
        plt.title(f"Pass@Tokens\n{label}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_path = graph_dir / "pass_at_tokens.png"
        plt.savefig(out_path)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.plot(xs_pl, ys_pl, linestyle="-")
        plt.xlabel("Planner tokens budget")
        plt.ylabel("Pass@tokens")
        plt.ylim(0.5, 1.0)
        plt.xlim(0, 100000)
        plt.title(f"Pass@Tokens (Planner only)\n{label}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_path_pl = graph_dir / "pass_at_tokens_planner.png"
        plt.savefig(out_path_pl)
        plt.close()

    if has_time:
        planner_time = df["_planner_time_reward"].fillna(0).to_numpy(dtype=float)
        prover_time = df["_prover_time_reward"].fillna(0).to_numpy(dtype=float)
        total_time = planner_time + prover_time
        xs_t, ys_t = _compute_pass_curve(total_time, success)
        xs_t_pl, ys_t_pl = _compute_pass_curve(planner_time, success)

        plt.figure(figsize=(5, 4))
        plt.plot(xs_t, ys_t, linestyle="-")
        plt.xlabel("Total inference time budget (seconds)")
        plt.ylabel("Pass@time")
        plt.ylim(0.4, 1.0)
        plt.title(f"Pass@Time\n{label}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_path = graph_dir / "pass_at_time.png"
        plt.savefig(out_path)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.plot(xs_t_pl, ys_t_pl, linestyle="-")
        plt.xlabel("Planner inference time budget (seconds)")
        plt.ylabel("Pass@time")
        plt.ylim(0.4, 1.0)
        plt.title(f"Pass@Time (Planner only)\n{label}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_path_pl = graph_dir / "pass_at_time_planner.png"
        plt.savefig(out_path_pl)
        plt.close()
