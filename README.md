# Universal AI Inference Runtime (UAII)

**Any model → UAII IR → any hardware.**

UAII is a modular C++ inference runtime: load a model into a common intermediate representation, optimize and schedule it, then execute on CPU or GPU backends through one session API. It is built as an *execution platform*—loaders, operators, backends, schedulers, and storage are pluggable—not a single-family engine hard-wired to one format.

| | |
|---|---|
| **Language** | C++17 |
| **Build** | CMake 3.20+ |
| **Stable ABI** | C API `0.3.0` (`uaii_capi`) |
| **License** | [MIT](LICENSE) |

---

## What it does

1. **Ingest** models from GGUF, Safetensors, ONNX, MLX (weights + config), or PyTorch export sidecars into **UAII IR**.
2. **Validate & plan** the graph (shapes, dtypes, fusion, memory reuse, optional disk plan cache).
3. **Execute** via a `Session` on a chosen backend, with competitive CPU GEMM, in-memory quantized MatMul, KV-cache generation for Llama-family GGUF, and optional weight streaming.
4. **Integrate** through the `uaii` CLI, the C ABI shared library, or the Python SDK.

```text
  GGUF / Safetensors / ONNX / MLX / PyTorch
                    │
                    ▼
              ┌──────────┐
              │  UAII IR │  validate · fuse · plan
              └────┬─────┘
                   │
                   ▼
              ┌──────────┐
              │ Session  │  KV cache · quant GEMM · streaming
              └────┬─────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
   CPU          CUDA*         Metal* / Vulkan* / WebGPU* / ROCm*
 (oneDNN /     cuBLASLt
  OpenBLAS /   + kernels
  tiled ref)
```

\*GPU paths require the matching `UAII_WITH_*=ON` build and a present device. Capability strings and `uaii doctor` report host-fallback honestly—there is no silent “GPU name, CPU math.”

---

## Features

### Model formats

| Format | Role |
|---|---|
| **GGUF** | Llama-family transformer import (`llama`, `llama3`, `mistral`, `qwen2`, `phi3`); full `blk.*` stack with RMSNorm, QKV, RoPE, Attention + KV, SwiGLU, lm_head |
| **Safetensors** | Weight graphs / HF-style layouts → UAII IR |
| **ONNX** | Import to IR (companion `.uaii.json` or ONNX proto when enabled) |
| **MLX** | Directory with `config.json` + `.safetensors` (weights + config, not the Apple MLX runtime) |
| **PyTorch** | `.pt` / `.pth` via exported `.onnx` or `.uaii.json` sidecar (`UAII_WITH_LIBTORCH` reserved for TorchScript) |

Convert anything the loader registry accepts:

```bash
uaii convert model.gguf -o model.uaii.json
uaii convert model.onnx -o model.uaii.json
```

### Quantization & compute

- **In-memory GGUF block quants** without full f32 unpack: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`–`Q6_K`
- Pack/unpack helpers: F16, BF16, INT8, INT4, NF4, MXFP4
- Session policy: `compute_dtype` (`F32` / `F16`), `keep_quantized_weights`
- Unsupported / IQ\* GGUF types **fail closed** (no silent Ones weights)

See [docs/gguf_support.md](docs/gguf_support.md).

### LLM runtime

- Prefill + greedy decode: `Session::generate` / `uaii_session_generate`
- First-class **KV cache** (per-layer K/V, context limit from metadata, fail-closed overflow)
- Long-sequence awareness; `UAII_MAX_LAYERS` optional layer cap
- Tokenizers: **BPE**, **SentencePiece** (`UAII_WITH_SENTENCEPIECE`), GGUF `tokenizer.ggml.*`, plus SimpleTokenizer for demos

### Hardware backends

| Backend | Without vendor SDK | With `UAII_WITH_*=ON` + device |
|---|---|---|
| **CPU** | Full f32 kernels; tiled/ref GEMM | + oneDNN / OpenBLAS when linked |
| **CUDA** | Host-fallback executable | Device memory, **cuBLASLt**, elementwise/norm kernels; Attention may host-fallback (advertised) |
| **Metal** | Host-fallback | Shared buffers; MatMul / Add / RMSNorm via runtime MSL |
| **Vulkan** | Host-fallback | Device buffers; Add via compute when available; honest caps for remaining ops |
| **WebGPU** | Host-fallback | Buffer path when headers present; compute still limited |
| **ROCm** | Host-fallback | HIP memory; MatMul (rocBLAS); Add / RMSNorm HIP kernels |

CPU GEMM provider is selected at runtime (`ref` / `onednn` / `openblas`); `uaii doctor` prints the active provider. Details: [docs/backend_support.md](docs/backend_support.md).

### Runtime & tooling

- Graph **validator**, JSON + binary IR (`.uaii.json` / `.uaii`)
- Planner: op fusion, memory reuse, storage plan, disk plan cache
- Weight **streaming** with double-buffer host staging; CUDA async H2D overlap when native
- Chrome-trace **profiler**, benchmark CLI, cache status/clear
- Plugin operator ABI (`Neg` example) and plugin discovery
- Fail-closed defaults (`weight_init=none`)

---

## Quick start

### Prerequisites

- C++17 compiler (MSVC 2019+, Clang 10+, or GCC 9+)
- CMake 3.20+
- Ninja (recommended) or your platform generator
- Git  

Optional: CUDA Toolkit, Vulkan SDK, ROCm, oneDNN, OpenBLAS, SentencePiece, Node.js (docs site), Python 3.10+ (SDK).

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel
```

On Windows with Ninja + MinGW/MSVC, the same commands apply; multi-config generators place binaries under `build/Release/`.

### Smoke the CLI

Paths below assume a single-config build (`build/uaii` or `build/libs/uaii-cli/uaii`). Adjust for your generator.

```bash
uaii version
uaii doctor
uaii doctor --load-plugins

# IR
uaii validate examples/ir/toy_mlp.uaii.json
uaii inspect examples/ir/toy_mlp.uaii.json
uaii graph examples/ir/toy_mlp.uaii.json --format text

# Execute
uaii run --demo toy_mlp
uaii run --demo tiny_block
uaii run --demo gguf
uaii run --demo parity

uaii run examples/ir/toy_mlp.uaii.json \
  --weight-init ones \
  --input x=1,2,3,4 \
  --output y_prob

# Tokenize
uaii tokenize encode hello world
uaii tokenize encode "hello" --bpe vocab.json --merges merges.txt

# Optimize / profile
uaii run --demo optimize
uaii profile --demo --output uaii_profile.json
uaii benchmark --demo
uaii cache status
```

### Tests

```bash
ctest --test-dir build -C Release --output-on-failure
```

---

## CLI reference

| Command | Description |
|---|---|
| `uaii doctor` | Environment, modules, GEMM provider, backends, plugins |
| `uaii validate <path>` | Validate a UAII IR graph |
| `uaii inspect <path>` | Tensors, nodes, metadata |
| `uaii graph <path> [--format text\|dot\|json\|plan]` | Dump or visualize the graph |
| `uaii convert <model> -o <out.uaii.json>` | GGUF / Safetensors / ONNX / MLX / PyTorch sidecar → IR |
| `uaii tokenize encode\|decode …` | Simple / BPE / SentencePiece / GGUF tokenizer |
| `uaii run …` | Run IR or built-in demos (`--backend`, `--weight-init`, …) |
| `uaii profile` | Chrome-trace JSON |
| `uaii benchmark` | Timing harness |
| `uaii cache` | Plan-cache status / clear |
| `uaii help` / `uaii version` | Help and version |

Global options: `--config <toml>`, `--log-level <level>`, `--no-color`, `--load-plugins`.

Built-in demos include `toy_mlp`, `tiny_block`, `gguf`, `safetensors`, `moe`, `parity`, `optimize`, `streaming`, `profile`, and `quant`.

---

## CMake options

| Option | Default | Meaning |
|---|---|---|
| `UAII_BUILD_TESTS` | `ON` | Unit / smoke tests |
| `UAII_BUILD_PLUGINS` | `ON` | Example plugins |
| `UAII_BUILD_PYTHON` | `OFF` | pybind11 extension |
| `UAII_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` / `/WX` |
| `UAII_WITH_ONEDNN` | `OFF` | Intel oneDNN CPU GEMM |
| `UAII_WITH_OPENBLAS` | `OFF` | OpenBLAS CPU GEMM |
| `UAII_WITH_CUDA` | `OFF` | CUDA device path (cuBLASLt + kernels) |
| `UAII_WITH_METAL` | `OFF` | Metal device path |
| `UAII_WITH_VULKAN` | `OFF` | Vulkan device path |
| `UAII_WITH_WEBGPU` | `OFF` | WebGPU device path |
| `UAII_WITH_ROCM` | `OFF` | ROCm / HIP device path |
| `UAII_WITH_SENTENCEPIECE` | `OFF` | SentencePiece tokenizer |
| `UAII_WITH_LIBTORCH` | `OFF` | LibTorch loader hook |
| `UAII_WITH_ONNX` | `ON` | ONNX loader features |

Useful environment variables: `UAII_LOG_LEVEL`, `UAII_NUM_THREADS`, `UAII_GEMM`, `UAII_MAX_LAYERS`, `UAII_CAPI_PATH`, `UAII_PLUGIN__DIRS`.

---

## C API

Shared library target: **`uaii_capi`**. Header: [`include/uaii/c_api/uaii.h`](include/uaii/c_api/uaii.h).

Current version: **0.3.0** (pre-1.0). Callers must set `uaii_session_options.struct_size` after `uaii_session_options_init`.

Highlights:

- Create / destroy sessions, set inputs, run, read outputs
- `uaii_session_generate` for token-in / token-out generation
- Options: backend, fusion, streaming, profiler, `compute_dtype`, `keep_quantized_weights`, `max_context`
- Fail-closed weight init by default

Stability rules: [docs/c_api_stability.md](docs/c_api_stability.md).

---

## Python SDK

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

python bindings/python/scripts/bundle_native.py --build-dir build
pip install -e bindings/python

python examples/python/load_run_profile.py
```

If the native library is not found automatically:

```bash
# Windows PowerShell
$env:UAII_CAPI_PATH = "C:\path\to\uaii_capi.dll"
```

Wheel builds are available via [`.github/workflows/wheels.yml`](.github/workflows/wheels.yml) (tag or manual dispatch). More detail: [bindings/python/README.md](bindings/python/README.md).

---

## Configuration

Default file: [`configs/uaii.toml`](configs/uaii.toml). Environment overlays override file keys (for example `UAII_LOG_LEVEL` → `log.level`).

---

## Plugins

Dynamic libraries export `uaii_plugin_get_info`, `uaii_plugin_init`, and `uaii_plugin_shutdown`. The host rejects mismatched `UAII_PLUGIN_ABI_VERSION`.

See [`include/uaii/c_api/plugin_abi.h`](include/uaii/c_api/plugin_abi.h) and [`plugins/example_probe`](plugins/example_probe) / [`plugins/example_op`](plugins/example_op).

---

## Repository layout

```text
include/uaii/           Public C++ headers and C ABI
libs/uaii-core/         Errors, logging, config, plugin host
libs/uaii-ir/           Graph, registry, validator, serialize, plan
libs/uaii-runtime/      Session, schedulers, demos
libs/uaii-kernels/      CPU kernels, IGemm, quant GEMM
libs/uaii-backends/     CPU + CUDA / Metal / Vulkan / WebGPU / ROCm
libs/uaii-loaders/      GGUF, Safetensors, ONNX, MLX, PyTorch
libs/uaii-tokenizers/   Simple, BPE, SentencePiece, GGUF wiring
libs/uaii-quant/        Quant formats and GGUF dequant
libs/uaii-memory/       Arena / pool / budget allocator
libs/uaii-storage/      File provider, mmap, streaming weights
libs/uaii-planner/      Fusion, memory/storage plans, cache
libs/uaii-profiler/     Chrome-trace profiler
libs/uaii-capi/         Stable C ABI shared library
libs/uaii-cli/          `uaii` command-line tool
bindings/python/        Python SDK
website/                Next.js documentation site
examples/               IR samples, models notes, Python examples
schemas/                FlatBuffers IR contract
configs/                Default TOML
docs/                   Architecture and support matrices
tests/                  Unit and smoke tests
plugins/                Example plugins
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module and IR design |
| [docs/gguf_support.md](docs/gguf_support.md) | GGUF arches, quants, generate |
| [docs/backend_support.md](docs/backend_support.md) | Backend capability matrix |
| [docs/c_api_stability.md](docs/c_api_stability.md) | C ABI versioning |
| [docs/vision.md](docs/vision.md) | Product mission |
| [docs/feature.md](docs/feature.md) | Feature catalog |
| [website/](website/) | Static docs site (`npm ci && npm run build`) |

---

## Contributing

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

---

## License

MIT © 2026 Arjun Shukla and contributors
