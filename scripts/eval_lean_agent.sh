#!/usr/bin/env bash
# Evaluator for lean_agent environment.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# --- Model / endpoint ---
MODEL="${PLANNER_MODEL:-gemini-3.0-flash-preview}"
BASE_URL="${PLANNER_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
VERIFICATION_URL="${VERIFICATION_URL:-https://containers.datacrunch.io/verification/api/}"

# --- Evaluation ---
NUM=6                           # number of problems
ROLLOUTS=1                      # rollouts per problem
CONC=6                          # concurrency

# --- Environment args ---
MAX_TURNS=3                    # main model turn budget
SUBAGENT_MAX_TURNS=10            # subagent turn budget
VERIFY_TIMEOUT=60               # Lean compile timeout (seconds)

# --- Sampling ---
TEMP=0.7

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

ENV_ARGS=$(cat <<JSON
{
  "verification_url": "$VERIFICATION_URL",
  "max_turns": $MAX_TURNS,
  "subagent_max_turns": $SUBAGENT_MAX_TURNS,
  "verify_timeout": $VERIFY_TIMEOUT
}
JSON
)

SAMP_ARGS=$(cat <<JSON
{
  "temperature": $TEMP,
  "stop": ["</lean_check>", "</prove>"]
}
JSON
)

uv run \
  vf-eval lean_agent \
  -b "$BASE_URL" \
  -k PLANNER_KEY \
  -m "$MODEL" \
  -n "$NUM" \
  -r "$ROLLOUTS" \
  -c "$CONC" \
  -s \
  -a "$ENV_ARGS" \
  -S "$SAMP_ARGS" \
  --verbose
