Lean-Agent: MiniF2F Environments (verifiers 0.1.4)

Layout
- src/: environments + eval scripts (this folder)
- kimina-lean-server/: Lean REPL backend (HTTP /api)
- scripts/: vLLM server launcher (OpenAI-compatible)

Setup
- Python 3.12 + uv
- In src/: uv sync
- Install env packages (either):
  - Workspace: uv sync (members listed in pyproject)
  - Or packages: uv run vf-install minif2f-tool -p environments/minif2f_tool; uv run vf-install minif2f-multiturn -p environments/minif2f_multiturn

Start servers
- vLLM (OpenAI API): scripts/run_vllm_server.sh (uses verifiers.inference.vllm_server)
  - Honors HF_HOME/TRANSFORMERS_CACHE/VLLM_DOWNLOAD_DIR (optional)
  - Example: CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_vllm_server.sh
- Kimina (Lean REPL):
  - cd kimina-lean-server && bash setup.sh && source "$HOME/.elan/env"
  - Warm start (optional):
    - export LEAN_SERVER_MAX_REPLS=32
    - export LEAN_SERVER_MAX_REPL_MEM=2G
    - export LEAN_SERVER_INIT_REPLS='{"import Mathlib\nimport Aesop": 8}'
  - UV_NO_SYNC=1 uv run --frozen python -m server

Evaluate
- Point vf-eval to vLLM and Kimina:
  - uv run --frozen vf-eval minif2f_tool \
      -b http://127.0.0.1:8001/v1 -k OPENAI_API_KEY \
      -m stepfun-ai/StepFun-Prover-Preview-7B \
      -n 1 -r 1 -c 4 \
      -a '{"backend_url":"http://127.0.0.1:8000/api","max_turns":2}' \
      -S '{"temperature":0.999,"top_p":0.95,"max_tokens":16384,"stop":["<REPL>"]}' \
      --verbose

Notes
- Env defaults: max_turns=5, sampling (temperature/top_p/max_tokens) mirrors the StepFun script.
- Verification: 10× retry with short backoff; errors surfaced in <REPL> without aborting runs.
- Tool uses per rollout ≈ max_turns − 1 (assistant turns only).




