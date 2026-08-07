# Changelog

All notable changes to Universal AI Inference (UAII) are documented here.

## [0.2.0] — 2026-08-08

### Sampling

- `SampleParams`: temperature, top-k, nucleus (top-p), repetition penalty, seed
- `Session::generate` samples from **logits** when non-greedy
- CLI: `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--seed`, `--stop` / `--stop-token-id`
- `uaii chat --jsonl` accepts the same fields; Operator UI + `/v1` pass them through

### MoE

- `MoEExpertsSwiGLU` IR op + CPU kernel (top-k weighted SwiGLU experts)
- GGUF import for Mixtral / Qwen2MoE-style tensors: `ffn_gate_inp`, `ffn_*_exps`, optional `*_shexp`
- Dense SwiGLU / GELU path unchanged; MoE no longer fail-closed

### Operator UI & docs

- Dashboard framed as **operator UI** (build runtime + Node console), not consumer download
- [TRY.md](TRY.md) — 5-minute path
- Bench CI artifacts + [docs/benchmarks.md](docs/benchmarks.md) rerun instructions

### Version

- CMake / package version **0.2.0**

## [0.1.0] — prior

Initial public runtime milestones: IR, GGUF dense transformers, CLI, C API, benches schema v3, dashboard e2e.
