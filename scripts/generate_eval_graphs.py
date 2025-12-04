from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from . import pass_at_prover_calls  # type: ignore[import-not-found]
from . import planner_turns_vs_acc  # type: ignore[import-not-found]
from . import token_time_vs_acc  # type: ignore[import-not-found]


def generate_for_run(output_dir: Path) -> None:
    """Generate all evaluation graphs for a single vf-eval run.

    Currently:
      - pass@#prover-calls curve (per-run)
      - pass@planner-turns curve (if planner metrics available)
      - token/time vs accuracy curves (where metrics are available)
    """
    pass_at_prover_calls.process_run(output_dir)
    planner_turns_vs_acc.generate_for_run(output_dir)
    token_time_vs_acc.generate_for_run(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation graphs for one or more vf-eval runs."
    )
    parser.add_argument(
        "output_dirs",
        type=str,
        nargs="+",
        help="One or more vf-eval output directories containing results.jsonl",
    )
    args = parser.parse_args()
    dirs: List[Path] = []
    for raw in args.output_dirs:
        out_dir = Path(raw)
        if not out_dir.is_dir():
            raise SystemExit(f"Output directory not found: {out_dir}")
        dirs.append(out_dir)
    for out_dir in dirs:
        generate_for_run(out_dir)
    if len(dirs) > 1:
        pass_at_prover_calls.combined_pass_at_calls(dirs)


if __name__ == "__main__":
    main()

