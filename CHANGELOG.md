# Changelog

All notable changes to Universal AI Inference (UAII) are documented here.

## [Unreleased]

### Phone preset

- `--preset phone`: ~2 GiB, one staging buffer, 4 threads, KV window 512, expert LRU 256 MiB. Phone clamps to half of free RAM. Laptop, desktop, and server keep their targets when free RAM covers them
- Background, cancel, and memory-warning are safe to call from the UI thread while `generate` / `run` is in progress. A memory warning drops KV without clearing the longrope scale length
- C API **0.4.0**: preset resolve, RSS / available bytes, cancel, background pause, memory-warning KV drop
- ARM NEON dot in the quant GEMM. Vulkan mapped f32 MatMul accepts `transpose_b`
- `examples/mobile` builds with the Android NDK or an iOS toolchain and selects the backend with `auto`. See [docs/mobile.md](docs/mobile.md)
- `uaii chat --help` lists the phone preset and its 512-token KV window

## [0.3.1] — 2026-08-15

### Device auto-select

- CLI `generate` / `chat` / `run` default `--backend auto`: CPU unless a native GPU backend is actually available (CUDA, then Metal, ROCm, Vulkan)
- Dashboard Settings default is `auto` (same rule). Pin `--backend cpu` or Settings → CPU to force host
- `uaii doctor` prints `recommended:` for the name auto would pick

### Hugging Face decoder-generic

- Any `*ForCausalLM` / `*LMHeadModel` (deny vision/audio/encoder/SSM/RWKV) instead of a closed class allowlist
- Layout auto-detect: Llama split QKV, fused `qkv_proj`/`W_pack`/GPT-NeoX, GPT-2 Conv1D + GELU + LayerNorm, Mixtral/Qwen2-MoE per-expert SwiGLU
- HF MoE configs accepted; experts are individual IR weights and page through the expert LRU
- HF DeepSeek MLA still fail-closed (use GGUF)

### Storage-first (kimi DNA)

- `generate` / `chat` default `--preset laptop` (~8 GiB, pin trunk, stream rest, KV window 2048)
- Storage plan never pins MoE experts; streaming loads `file#tensor` via the real GGUF/safetensors readers (not raw file offset 0)
- Expert LRU with a hot window so several expert tensors can be resident for one MoE op
- `MoEExpertsSwiGLU` listed arity (`x, probs, g0,u0,d0, …`) for HF Mixtral and GGUF packed banks split to `#e=N` slices
- Session pages only the routed top-k experts per token
- Sliding KV + attention sink so long chats cannot grow the cache without bound

---

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
