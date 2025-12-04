#!/usr/bin/env bash
# Prover-only benchmark for MiniF2F.
set -euo pipefail

# Load config from .env (keys, URLs, model settings)
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# --- Models / endpoints (from .env, with defaults) ---
PROVER_MODEL="${PROVER_MODEL:-deepseek-ai/DeepSeek-Prover-V2-7B}"
PROVER_BASE_URL="${PROVER_BASE_URL:-https://containers.datacrunch.io/deepseek-prover-v2-7b/v1}"
VERIFICATION_URL="${VERIFICATION_URL:-http://127.0.0.1:8000/api}"

# --- Evaluation ---
NUM=244
ROLLOUTS=64
CONC=122

# --- Prover sampling ---
PROVER_MAX_COMPLETION_TOKENS=30000

# --- Limits ---
VERIFY_TIMEOUT=90
MAX_PROVER_ATTEMPTS=48
MAX_PARALLEL_PROVER=244

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

ENV_ARGS=$(cat <<JSON
{
  "verification_url": "$VERIFICATION_URL",
  "verify_timeout": $VERIFY_TIMEOUT,
	  "max_prover_attempts": $MAX_PROVER_ATTEMPTS,
	  "max_parallel_prover": $MAX_PARALLEL_PROVER,
	  "prover_model": "$PROVER_MODEL",
	  "prover_base_url": "$PROVER_BASE_URL",
	  "prover_api_key_env": "PROVER_KEY",
  "prover_sampling": {
    "max_completion_tokens": $PROVER_MAX_COMPLETION_TOKENS
  }
}
JSON
)


# Run evaluation (PROVER_KEY must be set)
uv run --frozen \
  vf-eval minif2f_prover_bench \
  -b "$PROVER_BASE_URL" \
  -k PROVER_KEY \
  -m "$PROVER_MODEL" \
  -n "$NUM" \
  -r "$ROLLOUTS" \
  -c "$CONC" \
  -s \
  -a "$ENV_ARGS" \
  --verbose
