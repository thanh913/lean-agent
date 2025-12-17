#!/usr/bin/env bash
# Direct prover benchmark for MiniF2F (no planner/decomposition).
set -euo pipefail

# Load config from .env (keys, URLs, model settings)
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# --- Models / endpoints (from .env, with defaults) ---
PROVER_MODEL="${PROVER_MODEL:-deepseek-ai/DeepSeek-Prover-V2-7B}"
PROVER_BASE_URL="${PROVER_BASE_URL:-}"
VERIFICATION_URL="${VERIFICATION_URL:-http://127.0.0.1:8000/api}"

# --- Evaluation ---
NUM="${NUM:-244}"                   # number of problems to evaluate
MAX_ATTEMPTS="${MAX_ATTEMPTS:-16}"  # proof attempts per problem
STAGGER_DELAY="${STAGGER_DELAY:-60}"  # seconds between starting attempts
CONC="${CONC:-64}"                  # concurrent problems
MAX_PARALLEL_LLM="${MAX_PARALLEL_LLM:-256}"  # global cap on concurrent LLM calls

# --- Prover sampling ---
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-30000}"

# --- Verification ---
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-120}"

# --- Output ---
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/prover_direct_${TIMESTAMP}.json}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

echo "=== Prover Direct Benchmark ==="
echo "Model: $PROVER_MODEL"
echo "Base URL: ${PROVER_BASE_URL:-default}"
echo "Verification: $VERIFICATION_URL"
echo "Problems: $NUM"
echo "Max attempts: $MAX_ATTEMPTS"
echo "Stagger delay: ${STAGGER_DELAY}s"
echo "Concurrency: $CONC"
echo "Max parallel LLM: $MAX_PARALLEL_LLM"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Max tokens: $MAX_TOKENS"
echo "Output: $OUTPUT_FILE"
echo "==============================="
echo

# Build arguments
ARGS=(
    -n "$NUM"
    --max-attempts "$MAX_ATTEMPTS"
    --stagger-delay "$STAGGER_DELAY"
    -c "$CONC"
    --max-parallel-llm "$MAX_PARALLEL_LLM"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-tokens "$MAX_TOKENS"
    --verify-timeout "$VERIFY_TIMEOUT"
    --model "$PROVER_MODEL"
    -o "$OUTPUT_FILE"
    -v
)

# Add base URL if set
if [[ -n "$PROVER_BASE_URL" ]]; then
    ARGS+=(--base-url "$PROVER_BASE_URL")
fi

# Run the benchmark
cd "$ROOT_DIR/environments/minif2f_decompose"
uv run python prover_bench.py "${ARGS[@]}"

echo
echo "Results saved to: $OUTPUT_FILE"
