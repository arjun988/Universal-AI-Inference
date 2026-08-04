# Universal AI Inference Runtime — Architecture

**Product:** Universal AI Inference Runtime (UAII Runtime)  
**Document:** Architecture  
**Version:** 0.1  
**Status:** Draft

---

## 1. Purpose

This document describes the system architecture of **UAII Runtime**: module boundaries, data flow, UAII IR, plugin model, scheduling, memory/storage, and backend contracts.

It complements:

- [Vision](./vision.md) — why
- [Plan](./plan.md) — when / tech stack
- [Features](./feature.md) — what capabilities ship

---

## 2. Architectural Thesis

UAII Runtime is an **execution operating system** for inference:

- **Loaders** translate external artifacts into **UAII IR**
- **Optimizers / validators** refine and check the graph
- **Planners** produce an execution plan (compute + memory + storage)
- **Scheduler** places and orders work across devices and tiers
- **Backends + kernels** execute operators
- **Profiler** observes the whole pipeline

**No backend understands GGUF, Safetensors, ONNX, or MLX.**  
Everything only understands **UAII IR** (and the execution plan derived from it).

---

## 3. High-Level Architecture

```
                 +-----------------------+
                 |     Model Loader      |
                 +-----------------------+
                            |
                            v
                   UAII Intermediate
                   Representation (IR)
                            |
           +----------------+----------------+
           |                                 |
    Graph Optimizer                   Graph Validator
           |                                 |
           +----------------+----------------+
                            |
                     Execution Planner
                            |
           +----------------+----------------+
           |                                 |
      Memory Planner                  Storage Planner
           |                                 |
           +----------------+----------------+
                            |
                         Scheduler
                            |
       +---------+---------+---------+---------+
       |         |         |         |         |
      CPU      CUDA      Metal    Vulkan    WebGPU
```

### 3.1 End-to-End Data Flow

```
Model file (GGUF / Safetensors / ONNX / …)
        │
        ▼
  uaii-loaders  ──►  UAII IR Graph
        │
        ▼
  validate + optimize
        │
        ▼
  uaii-planner  ──►  ExecutionPlan
                     (ops order, kernel picks, memory layout, storage tiers)
        │
        ▼
  uaii-runtime session
        │
        ├── uaii-memory   (buffers, pools, budgets)
        ├── uaii-storage  (handles, mmap, stream, compress)
        ├── uaii-kernels  (portable math)
        └── uaii-backends (device dispatch)
        │
        ▼
  Outputs + profiler traces
```

---

## 4. Design Principles (Architectural Implications)

| Principle | Implication |
|---|---|
| Universal | Core depends on IR + registries, never on a model family |
| Hardware independent | Stable `Backend` interface / C ABI; capability queries |
| Storage first | `TensorHandle` may reference non-RAM locations; scheduler costs include IO |
| Extensible | Plugins for loaders, ops, backends, storage, schedulers, quant, profilers |
| Deterministic | Documented reduction order / seed / precision modes |
| Performance first | Hot path in-process; no mandatory RPC; arena allocation; fusion |

---

## 5. Core Modules

### 5.1 `uaii-core`

Foundation shared by all libraries.

**Contains:**

- Runtime configuration
- Shared utilities
- Error type hierarchy
- Logging setup helpers
- Plugin discovery & loading
- Versioning helpers (library, IR, ABI)
- Common IDs (tensor ids, node ids, device ids)

**Must not contain:** model math, vendor SDK calls, format parsers.

---

### 5.2 `uaii-ir`

Universal Intermediate Representation. **Everything eventually becomes UAII IR.**

**Status:** Phase 2 implemented (`include/uaii/ir`, `libs/uaii-ir`).

**Responsibilities:**

- Tensor definitions (dtype, shape, layout, device hint, storage hint)
- Operator definitions (name, version, attributes, schema)
- Execution graph (nodes, edges, control/data deps)
- Graph metadata (model name, producer, IR version, domain)
- Dependency representation
- Validation APIs (structural + dtype/shape checks)
- Serialization: FlatBuffers schema (`schemas/uaii_ir.fbs`) + native binary/JSON codecs
- Execution plan structures (topological, pre-scheduler)

**Conceptual stack:**

```
Graph
  └── Nodes
        └── Operators (+ attributes)
              └── Execution Plan (derived, not format-specific)
```

---

### 5.3 `uaii-loaders`

Converts external model formats into UAII IR.

**Supported (planned):**

| Format | Notes |
|---|---|
| GGUF | Weights + metadata common in local LLM tooling |
| Safetensors | Tensor store; pair with config/tokenizer plugins |
| ONNX | Graph import where ops map cleanly |
| PyTorch export | TorchScript / Export IR bridges as available |
| MLX | Apple MLX artifacts (as feasible) |
| Custom | Plugin loaders for research formats |

Loaders **only** produce IR (+ optional weight blobs / tensor packs). They do not execute inference.

---

### 5.4 `uaii-runtime`

Graph execution host.

**Status:** Phase 3 CPU session implemented (`runtime::Session`, `CpuScheduler`).

**Responsibilities:**

- Execution lifecycle (create / run / destroy)
- Session creation (bound plan + resources)
- Tensor allocation requests (via memory/storage)
- Kernel dispatch (via backends)
- Synchronization (host/device, streams, events)
- Error propagation and cancellation hooks

---

### 5.5 `uaii-memory`

Owns allocation strategy for **device-visible** and host buffers.

**Status:** Phase 3 implemented (arena, pool, budget, allocator).

**Features:**

- Arena allocation
- Tensor pools / reuse
- Memory budgeting
- (Later) Pinned host memory, huge pages, NUMA

Memory works with **handles** coordinated with `uaii-storage` when tensors are not resident.

---

### 5.6 `uaii-storage`

Treats storage as a **backend-like resource**.

**Support targets:**

- RAM
- Disk / NVMe
- Object storage (later)
- Memory mapping
- Streaming reads
- Compression
- Remote storage / remote memory
- Future distributed cache

**Key abstraction: `TensorHandle`**

A `TensorHandle` may resolve to:

| Location | Example |
|---|---|
| RAM | Device or host pointer |
| NVMe | File region + offset |
| Cloud | Object key + range |
| Compressed archive | Codec + window |
| Network | Remote memory protocol |
| Future pool | Tiered fabric memory |

The scheduler and storage planner decide **when** to stage handles into executable memory.

---

### 5.7 `uaii-planner`

Produces optimized execution plans from validated IR.

**Responsibilities:**

- Kernel selection (given backend capabilities)
- Execution ordering (legal schedules)
- Operator fusion opportunities
- Scheduling hints (device affinity, priority)
- Memory reuse planning
- Storage tier decisions / prefetch pipeline generation
- Plan caching (keyed by IR hash + config)

**Sub-planners:**

- **Memory planner** — lifetimes, aliasing, peak workspace
- **Storage planner** — residency, streaming windows, compression

---

### 5.8 `uaii-backends`

Hardware abstraction layer (HAL).

**Status:** Phase 5 — `CpuBackend` plus CUDA / Metal / Vulkan / WebGPU / ROCm backends.
GPU backends always support **host-fallback** execution (same CPU kernels); optional
`UAII_WITH_*` native scaffolds prove the dispatch wiring without requiring vendor SDKs.

Each backend implements common interfaces:

| Interface area | Examples |
|---|---|
| Tensor allocation | Device buffers, views |
| Kernel dispatch | Launch configs, bindings |
| Synchronization | Events, barriers, host wait |
| Memory transfer | H2D, D2H, D2D, peer |
| Capability query | dtypes, op support, limits |
| Profiling hooks | timestamps, ranges |
| Parity | `ParityPolicy` + cross-backend compare |

**Backends:** CPU, CUDA, Metal, ROCm, WebGPU, Vulkan (factory-selectable); future FPGA/ASIC.

---

### 5.9 `uaii-kernels`

Portable mathematical kernels (reference + optimized variants).

**Status:** Phase 3 CPU f32 kernels: MatMul, Softmax, LayerNorm, RMSNorm, Relu/Gelu/Silu, Add/Mul, Identity.

**Examples (later):** Attention fused, Conv, Pooling, RoPE, MoE, Sampling, TopK.

Kernels are selected by the CPU dispatcher today; registry-backed plugin kernels follow.

---

### 5.10 `uaii-profiler`

Integrated profiler—not an afterthought.

**Features:**

- Timeline (Chrome-trace compatible)
- Memory over time
- Bandwidth estimates
- Thread visualization
- Kernel timings
- IO / storage timings
- Storage statistics
- Execution graph visualization hooks

---

### 5.11 `uaii-cli`

Official CLI (`uaii`): `run`, `benchmark`, `inspect`, `validate`, `profile`, `graph`, `convert`, `cache`, `doctor`.

---

### 5.12 `uaii-sdk`

Language bindings over a stable **C API** plus idiomatic wrappers:

Python · C++ · Go · Node · Swift · Java · C

---

## 6. UAII IR Deep Dive

### 6.1 Design Goals

1. **Complete enough** to express modern LLM / multimodal inference graphs
2. **Stable enough** for plan caching and plugin ecosystems
3. **Neutral** — not a thin wrapper over one format’s quirks
4. **Validatable** before any backend runs

### 6.2 Primary Objects

| Object | Role |
|---|---|
| `Graph` | Top-level executable unit |
| `Node` | Single operator invocation |
| `Value` / `Tensor` | SSA-like or ID-based data edges |
| `OpSpec` | Name + version + attribute schema |
| `Attribute` | Typed constants (ints, floats, strings, arrays) |
| `Metadata` | Human/producer info; not required for exec |
| `WeightRef` | Link to external or embedded weight blob |

### 6.3 Invariants

- Every node’s inputs/outputs type-check under validator rules
- Cycles forbidden unless explicitly marked control regions (future)
- Dtypes and layouts are explicit (no silent defaults on hot path)
- Backend-specific attrs are namespaced and optional

### 6.4 IR vs Execution Plan

| UAII IR | Execution Plan |
|---|---|
| What to compute | How / where / when |
| Portable | Device- and host-specific |
| Produced by loaders | Produced by planner |
| Serializable artifact | Cacheable derived artifact |

---

## 7. Plugin System

Plugins must be loadable with **zero modification** to `uaii-core` beyond registration via discovery.

### 7.1 Plugin Categories

| Category | Role |
|---|---|
| Model Loader | External format → UAII IR |
| Tokenizer | Text ↔ token ids |
| Operator | New op schemas + kernels |
| Backend | Device HAL implementation |
| Storage Provider | New residency backends |
| Quantization | Encode/decode / pack formats |
| Scheduler | Placement policies |
| Profiler | Extra telemetry exporters |
| Optimizer | Graph rewrite passes |
| Visualization | UI / export tools |

### 7.2 Discovery

- Dynamic libraries in configured plugin directories
- Manifest (`uaii-plugin.toml` or embedded header) with name, version, ABI, capabilities
- Host checks `UAII_PLUGIN_ABI` before calling entrypoints

### 7.3 Operator Registry

Operators register dynamically:

```text
RegisterOperator(
    name: "Attention",
    version: semver_or_u32,
    implementation: KernelEntry,
    backend_tags: ["cpu", "cuda", ...]
)
```

Future architectures add operators (Transformer blocks, MoE, MLA, Mamba, RWKV, Hyena, …) without forking the core.

---

## 8. Scheduler Architecture

The scheduler decides **where**, **when**, and **how** an operator executes.

### 8.1 Possible Targets

- CPU
- GPU (CUDA / Metal / ROCm / Vulkan / WebGPU)
- NPU (future)
- Remote node (future)
- Disk-backed staging (storage-aware)
- Hybrid pipelines (overlap compute + IO)

### 8.2 Cost Model Inputs

| Signal | Use |
|---|---|
| Memory pressure | Evict / stream / recomputed tradeoffs |
| Bandwidth | Prefer fused / local tensors |
| Compute cost | Kernel/device selection |
| Latency | Interactive vs batch policies |
| Parallelism | Wavefront / pipeline depth |
| Storage tier latency | Prefetch horizons |

Schedulers are pluggable; the default scheduler ships in-tree.

---

## 9. Memory System Architecture

### 9.1 Goals

- Minimize allocations on the hot path
- Predict peak memory before run
- Reuse tensors aggressively
- Avoid fragmentation via arenas/pools
- Support pinned memory, huge pages, NUMA
- Coordinate with streaming from storage

### 9.2 Collaboration with Planner

```
IR + backend caps
      │
      ▼
Lifetime analysis ──► peak workspace estimate
      │
      ▼
reuse / alias assignments
      │
      ▼
session binds arenas / pools
```

---

## 10. Storage Engine Architecture

Storage is a **first-class scheduling resource**, not a load-time detail.

### 10.1 Responsibilities

- Resolve `TensorHandle`s
- Stage windows of weights/activations into memory
- Apply compression codecs
- Report IO stats to profiler
- Honor budgets (RAM cap, NVMe bandwidth fair share)

### 10.2 Streaming Execution Pattern

```
Storage Planner defines windows
        │
        ▼
Prefetch next window (async IO)
        │
        ▼
Scheduler overlaps with compute
        │
        ▼
Release / recycle previous window buffers
```

---

## 11. Backend Interface Contract

Every backend must implement (C++ abstract interface + mirrored C ABI):

1. **Device enumeration & init**
2. **Tensor allocation / free / view**
3. **Kernel dispatch**
4. **Synchronization**
5. **Memory transfer**
6. **Capability query** (ops, dtypes, limits, alignments)
7. **Profiling hooks**

Capability query drives planner kernel selection; unsupported ops fail at plan time with actionable errors (or CPU fallback if policy allows).

---

## 12. Quantization Architecture

Built-in dtype / pack support (planned):

FP32 · BF16 · FP16 · INT8 · INT4 · NF4 · MXFP4

Quantization is represented in IR (dtype + layout + scale/zero-point metadata) and implemented via:

- Core packing/unpacking utilities
- Kernel variants for packed compute
- **Plugin API** for future formats

---

## 13. Observability & Tooling Architecture

Built-in visualization surfaces (CLI + SDK export):

| Tool | Sees |
|---|---|
| Execution Graph Viewer | IR / planned graph |
| Memory Viewer | Arenas, peaks, reuse |
| Storage Viewer | Handles, tiers, IO |
| IR Inspector | Nodes, attrs, dtypes |
| Kernel Inspector | Selected impls, launches |
| Execution Timeline | Profiler traces |
| Profiler Dashboard | Aggregated metrics |

---

## 14. Threading & Process Model

| Concern | Approach |
|---|---|
| Default | Single process, multi-thread |
| CPU parallelism | Thread pool per session/config |
| GPU | Backend streams / command queues |
| Storage IO | Async runtime for prefetch only |
| Plugins | Loaded in-process (trusted); sandboxing is a future topic |

---

## 15. Security & Trust Boundaries (Initial)

Phase 1–7 assumption: **plugins are trusted code** (same as loading a native `.so`).

Future work (post-MVP):

- Plugin signature verification
- Optional out-of-process plugin hosts
- Capability-limited storage providers

---

## 16. Tech Stack Anchors (Architecture View)

| Layer | Technology |
|---|---|
| Core orchestration | C++17 |
| Plugin / SDK ABI | C |
| IR serialization | FlatBuffers |
| CLI / logging | C++ CLI + `uaii::log` |
| CPU SIMD | Intrinsics (AVX2/NEON) |
| CUDA / HIP / Metal | Native vendor languages via backends |
| Vulkan / WebGPU | Vulkan SDK / Dawn |
| Python bindings | pybind11 or nanobind |

Full stack detail: [Plan §2](./plan.md#2-tech-stack).

---

## 17. Non-Goals (Architectural)

The following are out of the runtime boundary:

- Training / autograd graphs
- Distributed training collectives as a product focus
- Multi-tenant cloud control planes
- Model registry / hosting product features

Distributed **inference** (remote node targets) may appear later as scheduler/backend plugins without changing the IR-centric design.

---

## 18. Evolution Rules

1. Prefer new plugins over core changes.
2. Breaking IR changes bump IR major; provide loaders/migrators when possible.
3. Backend-specific optimizations stay behind capability APIs.
4. Hot-path interfaces stay allocation-free after session init.
5. Document determinism mode whenever numerical behavior changes.

---

## 19. Related Documents

- [Vision](./vision.md)
- [Plan](./plan.md)
- [Features](./feature.md)
