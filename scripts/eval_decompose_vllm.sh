#!/usr/bin/env bash
# Minimal evaluator for MiniF2F DecomposeEnv.
# Edit the variables below instead of passing CLI args.
set -euo pipefail

# Load config from .env (keys, URLs, model settings)
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# --- Models / endpoints (from .env, with defaults) ---
PLANNER_MODEL="${PLANNER_MODEL:-gemini-2.5-flash}"
PLANNER_BASE_URL="${PLANNER_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
PROVER_MODEL="${PROVER_MODEL:-deepseek-ai/DeepSeek-Prover-V2-7B}"
PROVER_BASE_URL="${PROVER_BASE_URL:-https://containers.datacrunch.io/deepseek-prover-v2-7b/v1}"
VERIFICATION_URL="${VERIFICATION_URL:-http://127.0.0.1:8000/api}"

# --- Evaluation ---
NUM=1                           # number of problems
ROLLOUTS=1                       # rollouts per problem
CONC=1                          # concurrency

# --- Prover sampling (OpenAI-compatible) ---
PROVER_MAX_COMPLETION_TOKENS=30000

# --- Environment args ---
PLANNER_BUDGET=1                # planner attempts per problem (minimal)
VERIFY_TIMEOUT=90               # Lean compile timeout (seconds)
MAX_PROVER_ATTEMPTS=4           # direct attempts per subgoal
MAX_PARALLEL_PROVER=310         # global cap on concurrent prover calls

# --- Planner sampling (planner only; prover sampling is hard-coded in env) ---
PLANNER_TEMP=0.7
PLANNER_TOP_P=0.9
PLANNER_MAX_TOKENS=200000

# Move to src/
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Build env args JSON
ENV_ARGS=$(cat <<JSON
{
  "verification_url": "$VERIFICATION_URL",
  "planner_budget": $PLANNER_BUDGET,
  "verify_timeout": $VERIFY_TIMEOUT,
  "max_prover_attempts": $MAX_PROVER_ATTEMPTS,
  "max_parallel_prover": $MAX_PARALLEL_PROVER,
  "prover_model": "$PROVER_MODEL",
  "prover_base_url": "$PROVER_BASE_URL",
  "prover_api_key_env": "PROVER_KEY",
  "shuffle": true,
  "shuffle_seed": 42,
  "prover_sampling": {
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
