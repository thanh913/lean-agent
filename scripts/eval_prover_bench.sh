#!/usr/bin/env bash
# Prover-only benchmark for MiniF2F.
set -euo pipefail

# --- Models / endpoints (OpenAI-compatible) ---
# PROVER_MODEL="Goedel-LM/Goedel-Prover-V2-8B"
PROVER_MODEL="deepseek-ai/DeepSeek-Prover-V2-7B"
PROVER_BASE_URL="https://containers.datacrunch.io/deepseek-prover-v2-7b/v1"

# --- Evaluation ---
NUM=1
ROLLOUTS=1
CONC=1
BACKEND_URL="http://127.0.0.1:8000/api"

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
  "backend_url": "$BACKEND_URL",
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
