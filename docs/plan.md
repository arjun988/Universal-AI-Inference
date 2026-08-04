# Universal AI Inference Runtime — Product & Development Plan

**Product:** Universal AI Inference Runtime (UAII Runtime)  
**Document:** Plan & Roadmap  
**Version:** 0.1  
**Status:** Draft

---

## 1. Overview

This plan turns the [Vision](./vision.md) into an executable roadmap: tech stack choices, repository layout, phased delivery, milestones, and success metrics.

**Goal chain:**

```
Any Model  →  UAII IR  →  Planner / Scheduler  →  Any Hardware Backend
```

---

## 2. Tech Stack

### 2.1 Core Runtime Language

| Layer | Choice | Rationale |
|---|---|---|
| **Primary language** | **C++17** (C++20 features optional later) | Industry standard for inference runtimes; zero-cost abstractions; excellent CUDA/Metal/HIP interop; wide contributor base |
| **Stable ABI / plugins** | **C ABI** (`extern "C"`) + versioned structs | Language-agnostic plugins; loaders/backends can be C++, C, or other languages via FFI |
| **GPU / vendor kernels** | **CUDA C++ / HIP / Metal Shading Language / WGSL** as needed | Meet vendor toolchains where they are; invoke via backend libraries |
| **Optional hot kernels** | Hand-tuned **SIMD** (AVX2/NEON intrinsics, optional ISPC later) | Performance-first CPU path |
| **Python** | Bindings / SDK / tooling (Phase 7; pybind11 or nanobind) | Ergonomic API for researchers; not used in the hot path |

C++ owns orchestration, IR, planning, memory, storage, CLI, and plugin host. Device-specific code lives behind backend interfaces. Python is a first-class SDK language, not the core runtime.

### 2.2 Build & Workspace

| Concern | Choice |
|---|---|
| Build system | **CMake 3.20+** (presets optional) |
| Layout | Modular `libs/uaii-*` libraries + shared `include/uaii` |
| Cross-compilation | CMake toolchains for aarch64 / secondary targets |
| Feature flags | CMake options per backend (`UAII_WITH_CUDA`, `UAII_WITH_METAL`, …) |
| CI | **GitHub Actions** — configure, build, test, clang-format/tidy checks |
| Packaging | CMake install + CPack; GitHub Releases for CLI; Python wheels later |
| Style | `.clang-format`, `.clang-tidy` |

### 2.3 Intermediate Representation & Serialization

| Concern | Choice |
|---|---|
| In-memory IR | Typed C++ graph (`uaii-ir`) |
| On-disk / IPC IR | **FlatBuffers** (primary) + optional JSON for debug |
| Schema evolution | Explicit IR version + compatibility policy |
| Hashing / cache keys | BLAKE3 (or XXH3) of canonical IR + planner config |

### 2.4 Memory, Storage & Concurrency

| Concern | Choice |
|---|---|
| Allocator | Custom arenas / pools in `uaii-memory` (over system allocator) |
| Async IO (storage) | Platform async IO / thread-pool prefetch (not on kernel hot path) |
| mmap | POSIX `mmap` / Win32 `MapViewOfFile` |
| Compression | `zstd`, optional `lz4` (vendored or system) |
| Object storage (later) | Abstract client interface (S3-compatible) |
| Threading | `std::thread` / thread pool; backend-owned streams/queues for GPU |

### 2.5 Backends (Planned Stack)

| Backend | Tech |
|---|---|
| CPU | C++ + SIMD; optional OpenMP |
| CUDA | CUDA Toolkit, optional NVRTC, cuBLAS where beneficial (behind interface) |
| Metal | Metal Performance Shaders / custom Metal shaders via ObjC++/FFI |
| ROCm | HIP |
| Vulkan | Vulkan SDK + compute pipelines |
| WebGPU | Dawn or wgpu-native via C API |

### 2.6 Tooling, CLI & SDKs

| Component | Stack |
|---|---|
| CLI | C++ (`uaii-cli`) |
| Config | TOML subset + env overlays (`uaii.toml`, `UAII_*`) |
| Errors | Structured `uaii::Error` / status codes |
| Logging | Structured severity logger (`uaii::log`) |
| Profiler export | Chrome trace JSON, optional Parquet later |
| Python SDK | **pybind11** or **nanobind** (Phase 7) |
| Node / Go / Java / Swift | C ABI bindings + thin idiomatic wrappers |
| Docs site | Docusaurus or MkDocs (later) |

### 2.7 Testing & Benchmarks

| Concern | Choice |
|---|---|
| Unit / integration | **GoogleTest** (fetched or system; enabled via CMake) |
| Property / fuzz | libFuzzer / AFL++ on IR validators (later) |
| Golden outputs | Determinism suites (bit-exact where required) |
| Benchmarks | Google Benchmark + custom `uaii benchmark` CLI |
| Model fixtures | Small GGUF / Safetensors samples in `tests/fixtures` (LFS if large) |

### 2.8 Tech Stack Summary Diagram

```
+-------------------------------------------------------------+
|  SDKs: Python (pybind11/nanobind) · C API · Go/Node/Swift   |
+-------------------------------------------------------------+
|  uaii-cli  (C++)                                            |
+-------------------------------------------------------------+
|  uaii-runtime · uaii-planner · uaii-profiler                |
|  uaii-memory · uaii-storage · uaii-kernels                  |
+-------------------------------------------------------------+
|  uaii-ir  (FlatBuffers schema) · uaii-loaders               |
+-------------------------------------------------------------+
|  Plugin Host (C ABI) <- loaders · ops · backends · storage  |
+----------+----------+----------+----------+-----------------+
|   CPU    |   CUDA   |  Metal   | Vulkan   | WebGPU / ROCm   |
|  SIMD    |  CUDA C++|  MSL     |  Vulkan  |  Dawn / HIP     |
+----------+----------+----------+----------+-----------------+
```

---

## 3. Repository Layout

```
Universal-AI-Inference/          # monorepo root (UAII Runtime)
├── docs/
│   ├── vision.md
│   ├── plan.md
│   ├── architecture.md
│   ├── feature.md
│   ├── architecture/            # deep-dive ADRs (future)
│   ├── design/
│   ├── research/
│   └── roadmap/
├── include/uaii/                # public headers
├── libs/
│   ├── uaii-core/
│   ├── uaii-ir/
│   ├── uaii-runtime/
│   ├── uaii-memory/
│   ├── uaii-storage/
│   ├── uaii-loaders/
│   ├── uaii-planner/
│   ├── uaii-backends/
│   ├── uaii-kernels/
│   ├── uaii-profiler/
│   └── uaii-cli/
├── plugins/                     # example / out-of-tree style plugins
├── bindings/
│   ├── python/                  # Phase 7
│   └── ...
├── cmake/
├── benchmarks/
├── examples/
├── tests/
├── tools/
├── CMakeLists.txt
└── README.md
```

Module responsibilities match [Architecture](./architecture.md).

---

## 4. Development Phases

### Phase 1 — Foundation (Weeks 1–4)

**Objective:** Scaffold the execution OS skeleton.

Deliverables:

- CMake workspace and `libs/uaii-*` module boundaries
- Build system + CI (configure/build/test, format checks)
- Plugin architecture (discovery, versioning, C ABI stubs)
- Logging, configuration (TOML subset/env), error system
- Core interfaces (backend, loader, operator, storage, scheduler)
- Initial documentation (this set)

**Exit criteria:** Empty plugin loads; `uaii doctor` reports environment; interfaces compile and are documented.

---

### Phase 2 — UAII IR (Weeks 5–8) — **Implemented**

**Objective:** Make the IR the single source of truth.

Deliverables:

- Graph, tensor, and operator definitions
- Operator registry (dynamic registration)
- Graph validator
- Serialization / deserialization (FlatBuffers schema + native binary/JSON codecs)
- Execution plan data structures (pre-scheduler)
- IR versioning rules

**Exit criteria:** Hand-authored IR graphs validate, serialize, round-trip, and dump via `uaii inspect` / `uaii graph`.

See [roadmap/PHASE2.md](./roadmap/PHASE2.md).

---

### Phase 3 — CPU Runtime (Weeks 9–14) — **Implemented**

**Objective:** First end-to-end inference on CPU.

Deliverables:

- Memory allocator (arenas, tensor pools, budgets)
- Scheduler (CPU placement, ordering)
- Execution engine lifecycle (session, dispatch, sync)
- CPU backend
- Basic kernels: MatMul, Softmax, LayerNorm / RMSNorm
- Minimal transformer-style graph execution

**Exit criteria:** Toy or small real model runs end-to-end on CPU through UAII IR with correct outputs.

See [roadmap/PHASE3.md](./roadmap/PHASE3.md).

---

### Phase 4 — Model Support (Weeks 15–20) — **Implemented**

**Objective:** Real formats and architectures enter via loaders.

Deliverables:

- GGUF loader → UAII IR
- Safetensors loader → UAII IR
- Tokenizer interface (plugin)
- Transformer operator set (attention, RoPE, MLP, etc.)
- Initial MoE support (routing + expert dispatch)

**Exit criteria:** At least one GGUF and one Safetensors model path produce usable generations; MoE smoke test passes.

See [roadmap/PHASE4.md](./roadmap/PHASE4.md).

---

### Phase 5 — Hardware Expansion

**Objective:** Prove hardware independence.

Deliverables (priority order):

1. CUDA backend
2. Metal backend
3. Vulkan backend
4. WebGPU backend
5. ROCm (as resources allow)

**Exit criteria:** Same UAII IR graph executes on ≥2 backends with validated numerical parity policy.

See [roadmap/PHASE5.md](./roadmap/PHASE5.md).

---

### Phase 6 — Optimization

**Objective:** Competitive performance and storage-aware execution.

Deliverables:

- Graph fusion passes
- Memory planner (reuse, lifetime analysis)
- Storage planner (tiering, streaming, mmap)
- Quantization pipeline (FP16/BF16/INT8/INT4/NF4/MXFP4 + plugin API)
- Streaming execution for oversized weights
- Integrated profiler + visualization hooks

**Exit criteria:** Measurable wins vs. Phase 3 baseline; profiler timelines for kernels/IO; streaming path for model larger than RAM (controlled fixture).

See [roadmap/PHASE6.md](./roadmap/PHASE6.md).

---

### Phase 7 — Ecosystem

**Objective:** Make UAII Runtime adoptable.

Deliverables:

- Python SDK (pybind11 + ctypes over stable C API)
- C API stability guarantees (semver)
- Documentation site (Next.js static export — no backend)
- Examples, benchmarks
- Design notes for future plugin marketplace

**Exit criteria:** External developer can load a model, run inference, and profile from Python without reading library internals.

See [roadmap/PHASE7.md](./roadmap/PHASE7.md).

---

## 5. Milestone Map

| Milestone | Phase | Signal |
|---|---|---|
| M0 Scaffold | 1 | Workspace + plugin host |
| M1 IR Complete | 2 | Validate + serialize IR |
| M2 CPU E2E | 3 | First inference |
| M3 Formats | 4 | GGUF + Safetensors |
| M4 Multi-Backend | 5 | CPU + one GPU backend |
| M5 Perf & Storage | 6 | Fusion + streaming + profile |
| M6 SDK | 7 | Python + C API GA-ready |

---

## 6. CLI Command Plan

Ship with `uaii-cli` (command prefix: `uaii`):

| Command | Purpose | Earliest phase |
|---|---|---|
| `uaii run` | Execute a model / IR graph | 3 |
| `uaii benchmark` | Throughput / latency suites | 3–6 |
| `uaii inspect` | Tensors, metadata, weights summary | 2 |
| `uaii validate` | IR + graph checks | 2 |
| `uaii profile` | Capture timelines | 6 |
| `uaii graph` | Dump / visualize execution graph | 2–6 |
| `uaii convert` | Format → UAII IR | 4 |
| `uaii cache` | Manage weight / plan caches | 6 |
| `uaii doctor` | Env, backends, drivers | 1 |

---

## 7. Risk Register & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Scope explosion across backends | Delay | CPU-first; feature-flag backends; shared IR contracts |
| Numerical mismatch across devices | Trust | Explicit determinism modes; golden tests; tolerance policies |
| Plugin ABI breakage | Ecosystem churn | Versioned C ABI; compatibility tests in CI |
| Storage-first complexity early | Slow Phase 3 | Start with RAM + mmap; tiering in Phase 6 |
| Competing with mature engines | Adoption | Differentiate on universality + storage + plugins, not day-1 peak tokens/s |

---

## 8. Success Metrics

1. **Format universality** — Run multiple model formats through a single runtime without changing the core.
2. **Plugin extensibility** — Add a new operator or backend by implementing a plugin, not modifying existing modules.
3. **CPU competitiveness** — Achieve competitive CPU performance with established runtimes on supported models (publish benchmark methodology).
4. **Observability** — Built-in profiling and execution visualization as first-class features.
5. **Architectural stability** — Maintain a stable core while supporting new model families over time (semver + IR version policy).

---

## 9. Versioning Policy (Initial)

| Artifact | Scheme |
|---|---|
| Libraries / CLI | SemVer |
| UAII IR | `uaii_ir_major.minor` with documented compatibility |
| Plugin ABI | `UAII_PLUGIN_ABI` integer; host rejects incompatible majors |
| Quant / op schemas | Namespaced version strings in registry |

---

## 10. Near-Term Working Agreement

1. Core must not import model-specific code.
2. All user-visible formats convert to UAII IR before execution.
3. Backends understand only UAII IR + planned kernels—not GGUF/Safetensors.
4. Performance changes require benchmarks; correctness changes require golden tests.
5. Public docs use **Universal AI Inference Runtime** / **UAII Runtime** only.

---

## 11. Related Documents

- [Vision](./vision.md)
- [Architecture](./architecture.md)
- [Features](./feature.md)
