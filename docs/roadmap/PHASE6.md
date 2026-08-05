# Phase 6 — Optimization

**Status:** Core planner/profiler/quant/streaming path is in-tree; performance is **not** production-competitive yet.

## Delivered

| Item | Reality |
|---|---|
| Graph fusion | Identity elim + MatMul→Relu → `MatMulRelu` (small surface) |
| Memory planner | Lifetime slot reuse |
| Storage planner | Tiering + streaming windows |
| Plan cache | In-memory + optional disk via `UAII_PLAN_CACHE_DIR` |
| Quantization | Pack/unpack helpers; **session compute remains f32** after dequant |
| Streaming weights | Double-buffer staging + prefetch; not full async pipeline |
| Profiler | Chrome Trace JSON |
| Threaded MatMul | `UAII_NUM_THREADS`; AVX2 path when compiled with AVX2 |
| OS mmap | `FileStorageProvider` uses MapViewOfFile / mmap |

## Partial / not done

- Broad SIMD/OpenMP coverage
- Competitive fused attention / GEMM
- Full IO/compute overlap with multiple outstanding transfers

## Verify locally

```bash
uaii run --demo optimize
uaii run --demo streaming
uaii run --demo profile
uaii run --demo quant --format int8
uaii benchmark --demo
```
