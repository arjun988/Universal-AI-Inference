# Phase 6 — Optimization (Complete in-tree)

**Status:** Implemented in repository (verify locally via README build steps)

## Delivered

| Item | Implementation |
|---|---|
| Graph fusion | `planner::apply_fusion_passes` (Identity elimination, MatMul→Relu → `MatMulRelu`) |
| Memory planner | `planner::build_memory_reuse_plan` (lifetimes + buffer slot reuse) |
| Storage planner | `planner::build_storage_plan` (tiering, mmap/disk, streaming windows) |
| Plan cache | `planner::PlanCache` + `uaii cache status\|clear` |
| Quantization | `uaii-quant` — F16/BF16/INT8/INT4/NF4/MXFP4 + `IQuantizer` plugin registry |
| Streaming weights | `storage::StreamingWeightStore` + session `--stream` under budget |
| Profiler | `profiler::Profiler` + Chrome Trace JSON (`uaii profile`) |
| Optimize pipeline | `planner::optimize_graph` wired into `Session::create` |
| Benchmark | `uaii benchmark --demo` (baseline vs optimized) |

## Exit criteria mapping

| Criterion | How to verify |
|---|---|
| Wins vs Phase 3 baseline | `uaii run --demo optimize` and `uaii benchmark --demo` |
| Profiler timelines | `uaii run --demo profile` / `uaii profile --demo` → `uaii_profile.json` |
| Streaming > RAM fixture | `uaii run --demo streaming` (staging < total weight bytes) |

## Verify locally

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel

uaii doctor
uaii run --demo optimize
uaii run --demo streaming
uaii run --demo profile
uaii run --demo quant --format int8
uaii run --demo quant --format nf4
uaii benchmark --demo
uaii profile --demo --output uaii_profile.json
uaii cache status
```
