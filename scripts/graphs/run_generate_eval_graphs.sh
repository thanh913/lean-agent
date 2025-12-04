#!/usr/bin/env bash
set -euo pipefail

# Edit these paths to point at vf-eval output directories that contain results.jsonl
# Example:
#   PROVER_BENCH_DIR="src/environments/minif2f_prover_bench/outputs/evals/minif2f_prover_bench--deepseek-ai--DeepSeek-Prover-V2-7B/59c7ac3b"
#   DECOMPOSE_DIR="src/environments/minif2f_decompose/outputs/evals/minif2f_decompose--gemini-2.5-pro/eccf8dad"

PROVER_BENCH_DIR="/mnt/block/lean-agent/src/environments/minif2f_prover_bench/outputs/evals/minif2f_prover_bench--deepseek-ai--DeepSeek-Prover-V2-7B/59c7ac3b"
DECOMPOSE_DIR="/mnt/block/lean-agent/src/environments/minif2f_decompose/outputs/evals/minif2f_decompose--gemini-2.5-pro/eccf8dad"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

uv run python -m scripts.generate_eval_graphs "$PROVER_BENCH_DIR" "$DECOMPOSE_DIR"

