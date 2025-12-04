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
NUM=244                             # number of problems to evaluate
ROLLOUTS=1                          # rollouts per problem
CONC=244                            # concurrent problems (vf-eval level)

# --- Prover sampling ---
PROVER_MAX_COMPLETION_TOKENS=30000

# --- Prover attempts (batch parallel) ---
# For each problem, fire MAX_PARALLEL_PROVER attempts in parallel per batch.
# Repeat batches until MAX_PROVER_ATTEMPTS exhausted or success.
# Example: 64 attempts with 8 parallel = 8 batches of 8 attempts each.
#          64 attempts with 64 parallel = 1 batch of 64 attempts (all at once).
VERIFY_TIMEOUT=90
MAX_PROVER_ATTEMPTS=128             # total attempts per problem
MAX_PARALLEL_PROVER=64              # attempts per batch (fires all at once)

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
