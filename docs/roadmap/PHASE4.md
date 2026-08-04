# Phase 4 — Model Support (Complete in-tree)

**Status:** Implemented in repository (verify locally via README build steps)

## Delivered

| Item | Implementation |
|---|---|
| GGUF → UAII IR | `loaders::GgufLoader` + `gguf_read/write/load_tensor_f32` |
| Safetensors → UAII IR | `loaders::SafetensorsLoader` + read/write/load |
| Convert CLI | `uaii convert <in> -o <out>` |
| Tokenizer | `tokenizers::SimpleTokenizer` (+ `uaii tokenize`) |
| Transformer ops | Embedding, RoPE, Attention, Reshape, Transpose (+ schemas) |
| MoE | `MoERouter` + `MoEExperts` kernels + smoke demo |
| Weight refs | `path#tensor` resolved from GGUF/Safetensors at session load |

## Verify locally

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel

uaii run --demo gguf
uaii run --demo safetensors
uaii run --demo moe

uaii tokenize encode hello world
uaii convert examples/models/tiny_demo.gguf -o examples/models/tiny_demo.uaii.json
```

Demos write tiny fixtures under `examples/models/` (or `uaii_phase4_models/`).
