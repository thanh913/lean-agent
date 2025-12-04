from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X_MAX_CALLS = 100


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


def _load_metadata(output_dir: Path) -> Dict[str, Any]:
    meta_path = output_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_graph_dir(output_dir: Path) -> Path:
    """Return src/graphs/<run_id>, where src is the repository src/ directory."""
    orig = output_dir.resolve()
    cur = orig
    src_root: Path | None = None
    while True:
        if cur.name == "src":
            src_root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if src_root is None:
        # Fallback: place graphs next to the output directory.
        graphs_root = orig.parent / "graphs"
    else:
        graphs_root = src_root / "graphs"
    run_id = orig.name
    run_graph_dir = graphs_root / run_id
    run_graph_dir.mkdir(parents=True, exist_ok=True)
    return run_graph_dir


def _compute_pass_at_calls(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "_final_verification_reward" not in df or "_prover_calls_reward" not in df:
        raise KeyError("results.jsonl missing required metrics for pass@calls")
    success = df["_final_verification_reward"].fillna(0).to_numpy(dtype=float)
    calls = df["_prover_calls_reward"].fillna(np.nan).to_numpy(dtype=float)
    n = len(success)
    if n == 0:
        raise ValueError("No rows to compute pass@calls")
    max_calls = int(np.nanmax(calls))
    if max_calls <= 0:
        # No prover calls at all; define trivial constant curve at accuracy.
        return np.array([0]), np.array([float(success.mean())])
    upper = min(max_calls, X_MAX_CALLS)
    if upper < 1:
        upper = 1
    bs = np.arange(1, upper + 1, dtype=int)
    pass_rates: List[float] = []
    for b in bs:
        solved_within_b = (success == 1.0) & (calls <= b)
        pass_b = float(solved_within_b.sum()) / float(n)
        pass_rates.append(pass_b)
    return bs, np.array(pass_rates, dtype=float)


def _label_from_meta(meta: Dict[str, Any]) -> str:
    env_id = meta.get("env_id") or "env"
    model = meta.get("model") or ""
    if model:
        return f"{env_id} ({model})"
    return str(env_id)


def compute_curve(output_dir: Path) -> tuple[np.ndarray, np.ndarray, str, Path]:
    """Return (xs, ys, label, per-run-graph-dir) for a single run."""
    df = _load_results(output_dir)
    meta = _load_metadata(output_dir)
    xs, ys = _compute_pass_at_calls(df)
    graph_dir = _ensure_graph_dir(output_dir)
    label = _label_from_meta(meta)
    return xs, ys, label, graph_dir


def process_run(output_dir: Path) -> None:
    xs, ys, label, graph_dir = compute_curve(output_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, linestyle="-")
    plt.xlabel("Prover call budget (B)")
    plt.ylabel("Pass@B")
    plt.ylim(0.4, 1.0)
    plt.xlim(0, X_MAX_CALLS)
    plt.title(f"Pass@Prover Calls\n{label}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path = graph_dir / "pass_at_prover_calls.png"
    plt.savefig(out_path)
    plt.close()


def combined_pass_at_calls(output_dirs: list[Path]) -> None:
    """Generate a combined pass@prover-calls plot for multiple runs."""
    if not output_dirs:
        return
    curves: list[tuple[np.ndarray, np.ndarray, str]] = []
    for out_dir in output_dirs:
        xs, ys, label, _ = compute_curve(out_dir)
        curves.append((xs, ys, label))
    # Determine graphs root (src/graphs) from the first run.
    first_graph_dir = _ensure_graph_dir(output_dirs[0])
    graphs_root = first_graph_dir.parent

    plt.figure(figsize=(6, 4))
    for xs, ys, label in curves:
        plt.plot(xs, ys, linestyle="-", label=label)
    plt.xlabel("Prover call budget (B)")
    plt.ylabel("Pass@B")
    plt.ylim(0.4, 1.0)
    plt.xlim(0, X_MAX_CALLS)
    plt.title("Pass@Prover Calls (comparison)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = graphs_root / "pass_at_prover_calls_combined.png"
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pass@#prover-calls curves for vf-eval output directories."
    )
    parser.add_argument(
        "output_dirs",
        type=str,
        nargs="+",
        help="One or more vf-eval output directories containing results.jsonl",
    )
    args = parser.parse_args()
    dirs: list[Path] = []
    for raw in args.output_dirs:
        out_dir = Path(raw)
        if not out_dir.is_dir():
            raise SystemExit(f"Output directory not found: {out_dir}")
        dirs.append(out_dir)
    for out_dir in dirs:
        process_run(out_dir)
    if len(dirs) > 1:
        combined_pass_at_calls(dirs)


if __name__ == "__main__":
    main()
