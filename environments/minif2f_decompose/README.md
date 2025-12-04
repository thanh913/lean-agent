# MiniF2F Decompose

This directory hosts the rewired MiniF2F planner environment that leverages the `defer` tactic rather than the legacy `prove_with` DSL.

Highlights:
- Planner emits Lean blueprints that may freely mix direct proofs with `defer` calls. The executor compiles the blueprint with an embedded `defer` prelude, collects any logged subgoals, calls the prover on each, and then re-verifies the final script once the proven bodies are spliced back.
- If a blueprint solves the theorem outright (no `defer` tasks), the executor simply re-checks the code without `sorry` and reports success.
- Planner transcripts are preserved verbatim; the environment only appends `<REPL>` messages summarizing executor outcomes plus the final proof block when available.

Key files:
- `env.py`: Drives planner turns, invokes the executor, and surfaces Lean feedback to the planner.
- `proof_executor.py`: Defines the `defer` prelude, parses task logs, runs prover calls, performs byte-accurate replacements, and handles final verification.
- `prover.py`: Async OpenAI-backed prover wrapper that returns Lean proof bodies.
- `lean_utils.py`: Helpers for fenced-block extraction, proof-body manipulation, and remote Lean compilation.

Configuration knobs (via env args or dataset info): `backend_url`, `planner_budget`, `verify_timeout`, `prover_model`, `prover_base_url`, `prover_sampling`, `max_parallel_prover`, `max_prover_attempts`, `prover_api_key_env`.
