#!/usr/bin/env bash
# Evaluator for MiniF2F DecomposeEnv - IMO problems only.
# Filters to imo_* and imosl_* problems (20 total).
set -euo pipefail

# Load config from .env (keys, URLs, model settings)
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# --- Models / endpoints (from .env, with defaults) ---
PLANNER_MODEL="${PLANNER_MODEL:-gemini-2.5-pro}"
PLANNER_BASE_URL="${PLANNER_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
PROVER_MODEL="${PROVER_MODEL:-deepseek-ai/DeepSeek-Prover-V2-7B}"
PROVER_BASE_URL="${PROVER_BASE_URL:-https://containers.datacrunch.io/deepseek-prover-v2-7b/v1}"
VERIFICATION_URL="${VERIFICATION_URL:-http://127.0.0.1:8000/api}"

# --- Evaluation (IMO subset: 20 problems) ---
NUM=20                           # all IMO problems
ROLLOUTS=1                      # rollouts per problem
CONC=20                          # concurrency

# --- Prover sampling (OpenAI-compatible) ---
PROVER_TEMP=0.7
PROVER_TOP_P=0.9
PROVER_MAX_COMPLETION_TOKENS=30000

# --- Environment args ---
PLANNER_BUDGET=24                # planner attempts per problem
VERIFY_TIMEOUT=90               # Lean compile timeout (seconds)
MAX_PROVER_ATTEMPTS=4          # direct attempts per subgoal
STAGGER_DELAY=20                # seconds between starting attempts
MAX_PARALLEL_PROVER=200         # global cap on concurrent prover calls

# --- Planner sampling ---
PLANNER_TEMP=0.7
PLANNER_TOP_P=0.9
PLANNER_MAX_TOKENS=200000

# Move to src/
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Build env args JSON (with IMO filter)
ENV_ARGS=$(cat <<JSON
{
  "verification_url": "$VERIFICATION_URL",
  "planner_budget": $PLANNER_BUDGET,
  "verify_timeout": $VERIFY_TIMEOUT,
  "max_prover_attempts": $MAX_PROVER_ATTEMPTS,
  "stagger_delay": $STAGGER_DELAY,
  "max_parallel_prover": $MAX_PARALLEL_PROVER,
  "prover_model": "$PROVER_MODEL",
  "prover_base_url": "$PROVER_BASE_URL",
  "prover_api_key_env": "PROVER_KEY",
  "shuffle": true,
  "shuffle_seed": 42,
  "filter_prefixes": ["imo_", "imosl_"],
  "prover_sampling": {
    "temperature": $PROVER_TEMP,
    "top_p": $PROVER_TOP_P,
    "max_completion_tokens": $PROVER_MAX_COMPLETION_TOKENS
  }
}
JSON
)

# Build sampling JSON for planner (vf-eval top-level)
SAMP_ARGS=$(cat <<JSON
{
  "temperature": $PLANNER_TEMP,
  "top_p": $PLANNER_TOP_P,
  "max_tokens": $PLANNER_MAX_TOKENS,
  "stop": ["</blueprint>"]
}
JSON
)

# Run evaluation (PLANNER_KEY and PROVER_KEY must be set)
uv run \
  vf-eval minif2f_decompose \
  -b "$PLANNER_BASE_URL" \
  -k PLANNER_KEY \
  -m "$PLANNER_MODEL" \
  -n "$NUM" \
  -r "$ROLLOUTS" \
  -c "$CONC" \
  -s \
  -a "$ENV_ARGS" \
  -S "$SAMP_ARGS" \
  --verbose
