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
| **Primary language** | **Rust** (Edition 2021+) | Zero-cost abstractions, strong ownership for memory/storage planners, excellent FFI, safe concurrency, modern packaging |
| **Stable ABI / plugins** | **C ABI** (`extern "C"`) + versioned structs | Language-agnostic plugins; loaders/backends can be Rust, C++, or C |
| **GPU / vendor kernels** | **CUDA C++ / HIP / Metal Shading Language / WGSL** as needed | Meet vendor toolchains where they are; invoke via backend crates |
| **Optional hot kernels** | Hand-tuned **SIMD** (Rust `std::arch`, or C++ with FFI) | Performance-first CPU path |

Rust owns orchestration, IR, planning, memory, storage, CLI, and plugin host. Device-specific code lives behind backend interfaces.

### 2.2 Build & Workspace

| Concern | Choice |
|---|---|
| Workspace | **Cargo workspace** (`uaii-*` crates) |
| Native / CUDA build glue | **CMake** (invoked from `build.rs` where required) |
| Cross-compilation | `cross` / target triples for aarch64, WASM (WebGPU path) |
| Feature flags | Cargo features per backend (`cuda`, `metal`, `vulkan`, `webgpu`, `rocm`) |
| CI | **GitHub Actions** — build, test, clippy, fmt, backend smoke jobs |
| Packaging | crates.io (libraries), GitHub Releases (CLI binaries), wheels via **maturin** (Python) |

### 2.3 Intermediate Representation & Serialization

| Concern | Choice |
|---|---|
| In-memory IR | Typed Rust graph (`uaii-ir`) |
| On-disk / IPC IR | **FlatBuffers** (primary) + optional JSON for debug |
| Schema evolution | Explicit IR version + compatibility policy |
| Hashing / cache keys | BLAKE3 of canonical IR + planner config |

### 2.4 Memory, Storage & Concurrency

| Concern | Choice |
|---|---|
| Allocator | Custom arenas / pools in `uaii-memory` (over system allocator) |
| Async IO (storage) | **Tokio** for storage planner / streaming (not on kernel hot path) |
| mmap | `memmap2` |
| Compression | `zstd`, optional `lz4` |
| Object storage (later) | AWS SDK / OpenDAL abstraction |
| Threading | Rayon for CPU parallel sections; backend-owned streams/queues for GPU |

### 2.5 Backends (Planned Stack)

| Backend | Tech |
|---|---|
| CPU | Rust + SIMD; optional OpenMP-style thread pools |
| CUDA | CUDA Toolkit, NVRTC optional, cuBLAS/cuDNN where beneficial (behind interface) |
| Metal | Metal Performance Shaders / custom Metal shaders via FFI |
| ROCm | HIP |
| Vulkan | `ash` + compute pipelines |
| WebGPU | `wgpu` |

### 2.6 Tooling, CLI & SDKs

| Component | Stack |
|---|---|
| CLI | Rust + **clap** + **tracing** / **tracing-subscriber** |
| Config | TOML + env overlays (`uaii.toml`, `UAII_*`) |
| Errors | `thiserror` / `anyhow` at boundaries |
| Logging | `tracing` with structured fields |
| Profiler export | Chrome trace JSON, optional Parquet later |
| Python SDK | **PyO3** + **maturin** |
| Node SDK | **napi-rs** (phase 7) |
| Go / Java / Swift | C ABI bindings + thin idiomatic wrappers |
| Docs site | mdBook or Docusaurus (later) |

### 2.7 Testing & Benchmarks

| Concern | Choice |
|---|---|
| Unit / integration | `cargo test`, crate-level tests |
| Property / fuzz | `proptest` / cargo-fuzz on IR validators |
| Golden outputs | Determinism suites (bit-exact where required) |
| Benchmarks | **Criterion** + custom `uaii benchmark` CLI |
| Model fixtures | Small GGUF / Safetensors samples in `tests/fixtures` (LFS if large) |

### 2.8 Tech Stack Summary Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  SDKs: Python (PyO3) · Rust · C API · Go/Node/Swift/Java   │
├─────────────────────────────────────────────────────────────┤
│  uaii-cli  (clap, tracing)                                  │
├─────────────────────────────────────────────────────────────┤
│  uaii-runtime · uaii-planner · uaii-profiler                │
│  uaii-memory · uaii-storage · uaii-kernels                  │
├─────────────────────────────────────────────────────────────┤
│  uaii-ir  (FlatBuffers schema) · uaii-loaders               │
├─────────────────────────────────────────────────────────────┤
│  Plugin Host (C ABI) ← loaders · ops · backends · storage   │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│   CPU    │   CUDA   │  Metal   │ Vulkan   │ WebGPU / ROCm   │
│  SIMD    │  CUDA C++│  MSL     │  ash     │  wgpu / HIP     │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

---

## 3. Repository Layout

```
uaii-runtime/                    # monorepo root (Universal AI Inference Runtime)
├── docs/
│   ├── vision.md
│   ├── plan.md
│   ├── architecture.md
│   ├── feature.md
│   ├── architecture/            # deep-dive ADRs (future)
│   ├── design/
│   ├── research/
│   └── roadmap/
├── crates/                      # or top-level crate dirs
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
│   ├── uaii-cli/
│   └── uaii-sdk/                # Rust SDK + C headers
├── bindings/
│   ├── python/
│   ├── node/                    # later
│   └── ...
├── benchmarks/
├── examples/
├── tests/
├── tools/
├── Cargo.toml                   # workspace
└── README.md
```

Module responsibilities match [Architecture](./architecture.md).

---

## 4. Development Phases

### Phase 1 — Foundation (Weeks 1–4)

**Objective:** Scaffold the execution OS skeleton.

Deliverables:

- Cargo workspace and crate boundaries
- Build system + CI (fmt, clippy, test)
- Plugin architecture (discovery, versioning, C ABI stubs)
- Logging (`tracing`), configuration (TOML/env), error system
- Core interfaces (backend, loader, operator, storage, scheduler)
- Initial documentation (this set)

**Exit criteria:** Empty plugin loads; `uaii doctor` reports environment; interfaces compile and are documented.

---

### Phase 2 — UAII IR (Weeks 5–8)

**Objective:** Make the IR the single source of truth.

Deliverables:

- Graph, tensor, and operator definitions
- Operator registry (dynamic registration)
- Graph validator
- Serialization / deserialization (FlatBuffers)
- Execution plan data structures (pre-scheduler)
- IR versioning rules

**Exit criteria:** Hand-authored IR graphs validate, serialize, round-trip, and dump via `uaii inspect` / `uaii graph`.

---

### Phase 3 — CPU Runtime (Weeks 9–14)

**Objective:** First end-to-end inference on CPU.

Deliverables:

- Memory allocator (arenas, tensor pools, budgets)
- Scheduler (CPU placement, ordering)
- Execution engine lifecycle (session, dispatch, sync)
- CPU backend
- Basic kernels: MatMul, Softmax, LayerNorm / RMSNorm
- Minimal transformer-style graph execution

**Exit criteria:** Toy or small real model runs end-to-end on CPU through UAII IR with correct outputs.

---

### Phase 4 — Model Support (Weeks 15–20)

**Objective:** Real formats and architectures enter via loaders.

Deliverables:

- GGUF loader → UAII IR
- Safetensors loader → UAII IR
- Tokenizer interface (plugin)
- Transformer operator set (attention, RoPE, MLP, etc.)
- Initial MoE support (routing + expert dispatch)

**Exit criteria:** At least one GGUF and one Safetensors model path produce usable generations; MoE smoke test passes.

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

---

### Phase 7 — Ecosystem

**Objective:** Make UAII Runtime adoptable.

Deliverables:

- Python SDK
- Idiomatic Rust SDK polish
- C API stability guarantees (semver)
- Documentation site, examples, benchmarks
- Design notes for future plugin marketplace

**Exit criteria:** External developer can load a model, run inference, and profile from Python without reading crate internals.

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
| Crates / CLI | SemVer |
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
