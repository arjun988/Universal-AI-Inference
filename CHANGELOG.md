# Changelog

All notable changes to Universal AI Inference (UAII) are documented here.

## [0.3.0] — 2026-08-14

### Hugging Face loader

- First-class HF **directory** ingest: `config.json` + tokenizer* + `*.safetensors` (± Hub shards / `model.safetensors.index.json`)
- Strict config parse + architecture allowlist; RoPE, GQA, tied embeddings on Llama-style layouts
- Docs: [docs/huggingface_support.md](docs/huggingface_support.md); fixture under `tests/fixtures/hf/tiny_llama/`
- Dashboard: list HF folders, Hub pull, import path, chat with `[HF]` models

### Memory & Hub

- Memory presets: `--preset laptop|desktop|server` (+ `--ram-gb`)
- Peak RSS helper (`process_peak_rss_bytes`) reported in generate / session stats
- `uaii pull <org/model>` — download Hub repos into a local directory
- Chunked **prefill reporting** (progress stride via `prefill_chunk_size`; not full multi-token chunk kernels yet)

### P2 — MLA / IQ / CUDA Attention / packaging

- **DeepSeek MLA (GGUF):** expand-then-attend into full K/V + existing KV cache (`mla=1`); compressed MLA cache still future
- **IQ4_NL** GGUF dequant + GEMM path; other IQ* fail closed with convert-to-Q4_K_M hint; `GgufType` aligned with llama.cpp
- **CUDA:** decode-path Attention + RoPE on GPU (staged H2D/D2H); `attention_host_fallback=false` when native CUDA inits; prefill Attention may still CPU
- **PyPI:** `bindings/python/scripts/build_wheel.*` + `bundle_native.py`; clearer ImportError; CI workflow `python-wheel.yml`
- **Signing:** [docs/signing.md](docs/signing.md) + `scripts/package_release.ps1` (unsigned zip — certs required for real signing)

### P3 — research scaffolds & serving queue (no marketplace)

- Plugin `research_ops`: SSMConv1d, SSMState, RWKVTimeMix, EncoderCrossAttn, VisionPatchEmbed (fail-closed scaffolds)
- Dashboard warm-chat **FIFO queue** (`UAII_CHAT_QUEUE_MAX`), HTTP **503** + `Retry-After` when full; `GET /api/chat/queue`

### Version

- CMake / package version **0.3.0**

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
