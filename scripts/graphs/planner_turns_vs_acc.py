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


def _compute_pass_curve(turns: np.ndarray, success: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pass@B for planner turns (integer budgets)."""
    n = len(success)
    if n == 0:
        raise ValueError("Empty dataset")
    turns = turns.astype(float)
    turns = turns[~np.isnan(turns)]
    if turns.size == 0:
        return np.array([0]), np.array([float(success.mean())])
    max_turns = int(np.max(turns))
    if max_turns <= 0:
        return np.array([0]), np.array([float(success.mean())])
    budgets = np.arange(1, max_turns + 1, dtype=int)
    pass_rates: List[float] = []
    for b in budgets:
        solved_within_b = (success == 1.0) & (turns <= b)
        pass_b = float(solved_within_b.sum()) / float(n)
        pass_rates.append(pass_b)
    return budgets, np.array(pass_rates, dtype=float)


def generate_for_run(output_dir: Path) -> None:
    """Generate pass@planner-turns curve for a single vf-eval run (if available)."""
    df = _load_results(output_dir)
    if "_planner_turns_reward" not in df or "_final_verification_reward" not in df:
        return

    turns = df["_planner_turns_reward"].fillna(0).to_numpy(dtype=float)
    success = df["_final_verification_reward"].fillna(0).to_numpy(dtype=float)
    xs, ys = _compute_pass_curve(turns, success)

    meta = _load_metadata(output_dir)
    label = _label_from_meta(meta)
    graph_dir = _ensure_graph_dir(output_dir)

    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, linestyle="-")
    plt.xlabel("Planner turn budget (B)")
    plt.ylabel("Pass@planner_turns")
    plt.ylim(0.4, 1.0)
    plt.title(f"Pass@Planner Turns\n{label}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path = graph_dir / "pass_at_planner_turns.png"
    plt.savefig(out_path)
    plt.close()
