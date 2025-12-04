# MiniF2F Prover Bench

Minimal environment that benchmarks the prover directly: no planner loop, just
prover calls + Lean verification on MiniF2F snippets.

- Reuses `OpenAIProver` from `minif2f_decompose` (same prompt/sampling logic).
- Inputs: `prover_model`, `prover_base_url`, `prover_sampling`,
  `max_parallel_prover`, `max_prover_attempts`, `verify_timeout`.
- Output: single `<REPL>` feedback summarizing prover success/failure and logs.
