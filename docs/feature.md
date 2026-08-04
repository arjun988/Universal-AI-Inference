# Universal AI Inference Runtime — Features

**Product:** Universal AI Inference Runtime (UAII Runtime)  
**Document:** Feature Specification  
**Version:** 0.1  
**Status:** Draft

---

## 1. Purpose

This document catalogs product features for **UAII Runtime**: what users and integrators can do, which modules provide each capability, and how features map to development phases.

Related docs: [Vision](./vision.md) · [Architecture](./architecture.md) · [Plan](./plan.md)

---

## 2. Feature Themes

| Theme | Promise |
|---|---|
| **Universal execution** | Any model family via UAII IR + plugins |
| **Hardware portability** | Same graph, multiple backends |
| **Storage-first inference** | Tiered residency, streaming, compression |
| **Extensibility** | Loaders, ops, backends, schedulers as plugins |
| **Determinism & correctness** | Validated graphs, golden outputs |
| **Performance** | Fusion, memory reuse, SIMD / GPU kernels |
| **Observability** | Profiler, graph/memory/storage viewers |
| **Developer experience** | CLI, SDKs, convert/inspect/doctor workflows |

---

## 3. Model Loading & Conversion

### 3.1 Format Support

| Feature | Description | Module | Phase |
|---|---|---|---|
| GGUF load | Parse GGUF into UAII IR + weight refs | `uaii-loaders` | 4 |
| Safetensors load | Map tensors + metadata into IR | `uaii-loaders` | 4 |
| ONNX import | Import ONNX graphs where ops map | `uaii-loaders` | 4+ |
| PyTorch export bridge | Import supported exported graphs | `uaii-loaders` | 4+ |
| MLX import | Import MLX artifacts when feasible | `uaii-loaders` | 5+ |
| Custom loader plugins | Community / research formats | plugins | 1 (API), 4+ |
| `uaii convert` | CLI conversion to UAII IR artifact | `uaii-cli` | 4 |

### 3.2 Conversion Guarantees

- Loaders never execute model math beyond optional shape inference helpers.
- Output is always versioned **UAII IR** (+ weight pack).
- Conversion errors are structured (missing tensor, unsupported op, dtype mismatch).

---

## 4. UAII IR Features

| Feature | Description | Phase |
|---|---|---|
| Tensor model | Shape, dtype, layout, storage hints | 2 |
| Operator model | Name, version, attributes, schemas | 2 |
| Graph model | Nodes, edges, metadata, weight refs | 2 |
| Graph validator | Structural + type/shape checks | 2 |
| Serialization | FlatBuffers primary; JSON debug dump | 2 |
| IR inspect | Human-readable inspection via CLI/SDK | 2 |
| IR versioning | Compatibility policy for caches/plugins | 2 |
| Plan structures | ExecutionPlan types derived from IR | 2–3 |

**Invariant:** Backends and kernels consume IR/plan only—not source formats.

---

## 5. Operator & Kernel Features

### 5.1 Operator Registry

| Feature | Description |
|---|---|
| Dynamic registration | `RegisterOperator(name, version, impl)` |
| Versioned ops | Multiple impls coexist; planner selects |
| Backend tagging | Ops declare supported backends |
| Schema validation | Attributes checked before plan/exec |

### 5.2 Core Operator Families (Planned)

| Family | Examples | Phase |
|---|---|---|
| Linear algebra | MatMul, GEMM variants | 3 |
| Normalizations | LayerNorm, RMSNorm | 3 |
| Activations | GELU, SiLU, ReLU, … | 3 |
| Softmax / sampling | Softmax, TopK, sampling | 3–4 |
| Attention | SDPA / MHA / GQA-style Attention | 4 |
| Positional | RoPE | 4 |
| Conv / pool | Conv, Pooling | 4–5 |
| MoE | Router, expert gather/scatter | 4 |
| Research | MLA, Mamba, RWKV, Hyena via plugins | 4+ |

### 5.3 Kernel Delivery

| Feature | Description |
|---|---|
| Reference kernels | Correctness baseline (CPU) |
| Optimized CPU kernels | SIMD-first paths |
| Backend-specialized kernels | CUDA / Metal / … variants |
| Plugin kernels | Out-of-tree operator packs |

---

## 6. Runtime & Session Features

| Feature | Description | Module | Phase |
|---|---|---|---|
| Session API | Create session from plan + config | `uaii-runtime` | 3 |
| Lifecycle | Init → run → sync → teardown | `uaii-runtime` | 3 |
| Feed / fetch | Bind inputs/outputs by name or id | `uaii-runtime` | 3 |
| Batch execution | Configurable batch dimensions | `uaii-runtime` | 3–4 |
| Cancellation | Cooperative cancel between steps | `uaii-runtime` | 4+ |
| Multi-session | Concurrent sessions with budgets | `uaii-runtime` | 6 |
| Determinism modes | Strict vs fast numeric policies | `uaii-runtime` | 3–6 |

---

## 7. Planning Features

| Feature | Description | Phase |
|---|---|---|
| Kernel selection | Capability-aware choice of impl | 3–6 |
| Execution ordering | Legal topological / pipeline schedules | 3 |
| Operator fusion | Identity removal + MatMulRelu (`planner::apply_fusion_passes`) | 6 |
| Memory reuse plan | Lifetime-based buffer sharing | 6 |
| Storage plan | Tiering, prefetch windows | 6 |
| Scheduling hints | Device affinity, priorities | 5–6 |
| Plan cache | Cache by IR hash + config | 6 |
| Fallback policy | Optional CPU fallback for missing GPU ops | 5 |

---

## 8. Memory Features

| Feature | Description | Phase |
|---|---|---|
| Arena allocators | Fast bump / scoped arenas | 3 |
| Tensor pools | Reuse across steps | 3 |
| Memory budgeting | Hard/soft limits per session | 3 |
| Lifetime analysis | Peak and reuse reports | 6 |
| Pinned host memory | Faster H↔D transfers | 5 |
| Huge pages | Optional large-page backing | 6 |
| NUMA awareness | Node-local allocation policies | 6 |
| Fragmentation control | Pool size classes / arenas | 3–6 |

---

## 9. Storage Features

| Feature | Description | Phase |
|---|---|---|
| `TensorHandle` API | Location-agnostic tensor reference | 3–6 |
| RAM residency | Default hot tier | 3 |
| mmap files | Map weight files without full read | 4 |
| NVMe / disk tier | Spill / stream from local SSD | 6 |
| Streaming weights | Windowed load for large models | 6 |
| Compression | zstd/lz4 (and plugins) | 6 |
| Object storage | S3-compatible / remote blobs | 7+ |
| Remote memory | Future fabric / networked tiers | Future |
| Distributed cache | Future shared weight cache | Future |
| Storage stats | Bytes, hits, stall time in profiler | 6 |

---

## 10. Scheduler Features

| Feature | Description | Phase |
|---|---|---|
| Device placement | CPU / GPU / hybrid | 3–5 |
| Ordering | Respect deps; expose parallelism | 3 |
| Overlap compute & IO | Prefetch while executing | 6 |
| Memory-pressure policies | Evict / recompute / stream | 6 |
| Pluggable schedulers | Alternate policies via plugins | 6–7 |
| Future remote targets | Schedule to remote nodes | Future |

---

## 11. Backend Features

### 11.1 Common Backend Capabilities

Every backend exposes:

- Device enumeration & selection
- Tensor allocate / free / view
- Kernel dispatch
- Synchronization primitives
- Memory transfers
- Capability queries
- Profiling hooks

### 11.2 Backend Roadmap

| Backend | Key features | Phase |
|---|---|---|
| **CPU** | Host kernels, session default | 3 |
| **CUDA** | Host-fallback + optional native scaffold (`UAII_WITH_CUDA`) | 5 |
| **Metal** | Host-fallback + optional native scaffold | 5 |
| **Vulkan** | Host-fallback + optional native scaffold | 5 |
| **WebGPU** | Host-fallback + optional native scaffold | 5 |
| **ROCm** | Host-fallback + optional native scaffold | 5 |
| **FPGA / ASIC** | Vendor plugins | Future |

Parity: `ParityPolicy` (atol/rtol) + `uaii run --demo parity` compares the same IR across backends.

### 11.3 Capability Query Examples

- Supported dtypes and alignments
- Max workspace / max threads
- Operator coverage matrix
- Unified memory / peer access flags
- Determinism limitations

---

## 12. Quantization Features

| Feature | Description | Phase |
|---|---|---|
| FP32 / FP16 / BF16 | Core floating formats | 3–4 |
| INT8 / INT4 | Integer packed formats | 6 |
| NF4 | 4-bit NormalFloat-style packing | 6 |
| MXFP4 | Microscaling float packs | 6 |
| Scale / ZP metadata | IR-level quant parameters | 6 |
| Quant plugin API | Future formats without core forks | 6 |
| Mixed precision graphs | Per-tensor dtypes in one plan | 6 |

---

## 13. Tokenizer Features

| Feature | Description | Phase |
|---|---|---|
| Tokenizer interface | Stable encode/decode plugin API | 4 |
| In-tree adapters | Common LLM tokenizers as plugins | 4 |
| Decoupled from runtime | Not required for raw tensor runs | 4 |

---

## 14. Profiling & Visualization Features

| Feature | Description | Phase |
|---|---|---|
| Kernel timings | Per-op durations | 6 |
| Timeline export | Chrome-trace JSON | 6 |
| Memory timeline | Allocation peaks / reuse | 6 |
| IO / storage timings | Stall and throughput | 6 |
| Bandwidth estimates | Compute vs memory bound hints | 6 |
| Thread visualization | Host concurrency view | 6 |
| Execution graph view | IR / plan visualization | 2–6 |
| Memory viewer | Arena / pool inspection | 6 |
| Storage viewer | Handle residency map | 6 |
| IR inspector | Node/attr deep dive | 2 |
| Kernel inspector | Selected implementations | 6 |
| Profiler dashboard | Aggregated UX (CLI/SDK first) | 6–7 |

---

## 15. CLI Features

Command prefix: **`uaii`**

| Command | Feature | Phase |
|---|---|---|
| `uaii run` | Run model or IR with inputs | 3 |
| `uaii benchmark` | Latency/throughput suites | 3–6 |
| `uaii inspect` | Weights, tensors, metadata | 2 |
| `uaii validate` | IR / graph validation | 2 |
| `uaii profile` | Capture profiler session | 6 |
| `uaii graph` | Dump/visualize graph | 2–6 |
| `uaii convert` | Format → UAII IR | 4 |
| `uaii cache` | Manage IR/plan/weight caches | 6 |
| `uaii doctor` | Drivers, backends, env health | 1 |

CLI UX goals: actionable errors, JSON output mode for scripting, stable exit codes.

---

## 16. SDK Features

| Binding | Capabilities | Phase |
|---|---|---|
| **C API** | Session, tensor I/O, convert, profile (`uaii.h` 1.0.0) | 7 |
| **C++ SDK** | Idiomatic headers over `uaii-*` libraries | 3–7 |
| **Python** | `uaii.Session` load/run/profile (ctypes + optional pybind11) | 7 |
| **Docs site** | Next.js static export (`website/`, no backend) | 7 |
| **Go** | cgo / FFI wrapper | 7+ |
| **Node** | N-API bindings | 7+ |
| **Swift / Java** | Mobile/JVM wrappers | 7+ |

SDK requirements:

- Explicit resource lifetimes
- Error mapping to language idioms
- No need to understand library internals for common flows

---

## 17. Plugin System Features

| Feature | Description | Phase |
|---|---|---|
| Dynamic discovery | Load from plugin directories | 1 |
| ABI version gate | Reject incompatible plugins | 1 |
| Manifest metadata | Name, version, capabilities | 1 |
| Loader plugins | New formats | 1–4 |
| Operator plugins | New architectures | 2–4 |
| Backend plugins | New devices | 5 |
| Storage plugins | New tiers | 6 |
| Quant plugins | New numeric formats | 6 |
| Scheduler plugins | New policies | 6–7 |
| Optimizer plugins | Graph rewrites | 6 |
| Profiler / viz plugins | Exporters & UIs | 6–7 |

**Guarantee:** Adding a plugin does not require modifying `uaii-core` internals.

---

## 18. Configuration & Operability Features

| Feature | Description | Phase |
|---|---|---|
| TOML config | `uaii.toml` hierarchical config | 1 |
| Env overlays | `UAII_*` environment variables | 1 |
| Structured logging | `uaii::log` severity / fields | 1 |
| Error taxonomy | Stable error codes/categories | 1 |
| `uaii doctor` | Backend & driver diagnostics | 1 |
| Feature flags | Compile-time backend selection | 1–5 |
| Cache management | Disk caches for IR/plans | 6 |

---

## 19. Correctness & Quality Features

| Feature | Description | Phase |
|---|---|---|
| Graph validation | Fail fast before execution | 2 |
| Golden tests | Reference outputs for models/ops | 3+ |
| Determinism mode | Bit-identical where permitted | 3–6 |
| Parity tests | Cross-backend numerical policy | 5 |
| Fuzz / property tests | IR validator robustness | 2–4 |
| Benchmark gates | Perf regressions visible in CI | 6 |

---

## 20. Architecture Support Features (Model Families)

| Architecture | Support approach | Phase |
|---|---|---|
| Transformer (dense) | First-class ops + loaders | 4 |
| MoE | Routing + expert execution | 4 |
| MLA | Operator plugin / extension | 4+ |
| Mamba / SSM | Operator plugin | Future-ready |
| RWKV | Operator plugin | Future-ready |
| Hyena | Operator plugin | Future-ready |
| Custom research | Register ops + loader | Ongoing |

---

## 21. Explicit Non-Features

UAII Runtime does **not** include (see Vision non-goals):

- Training, fine-tuning, RLHF loops
- Dataset management
- Model hosting / hub product
- Full cloud serving control plane
- Experiment tracking

These may integrate **via SDKs** as external systems.

---

## 22. Feature → Success Metrics Mapping

| Success metric | Enabling features |
|---|---|
| Multiple formats, one core | Loaders → UAII IR → single runtime |
| Plugin-added ops/backends | Registry + HAL + ABI |
| Competitive CPU perf | SIMD kernels, fusion, memory planner |
| First-class profiling | `uaii-profiler` + CLI/SDK export |
| Stable core over time | IR versioning, plugin boundaries, semver |

---

## 23. MVP Feature Slice (Recommended)

Minimum compelling product after Phases 1–4:

1. Plugin host + `uaii doctor`
2. UAII IR validate / serialize / inspect
3. CPU session execution with MatMul, Softmax, norms
4. GGUF + Safetensors → IR
5. Transformer path + tokenizer interface
6. `uaii run` + `uaii convert` + `uaii validate`

Everything else (GPU matrix, fusion, streaming, SDKs) expands the same architecture without redesign.

---

## 24. Tech Stack (Feature Implementation Anchors)

| Feature area | Primary tech |
|---|---|
| Core features | C++17 libraries (`uaii-*`) |
| Plugins / SDK ABI | C ABI |
| IR exchange | FlatBuffers |
| CLI | C++ (`uaii-cli`) |
| Python features | pybind11 / nanobind |
| CPU accel | SIMD intrinsics |
| CUDA/Metal/ROCm/Vulkan/WebGPU | Backend-native stacks |
| Compression | zstd, lz4 |
| Async storage IO | Thread-pool / platform async (off hot path) |

See [Plan §2](./plan.md#2-tech-stack) for the full stack table.

---

## 25. Related Documents

- [Vision](./vision.md)
- [Architecture](./architecture.md)
- [Plan](./plan.md)
