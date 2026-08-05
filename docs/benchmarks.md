# Benchmarks

Reproducible microbenchmarks for UAII CPU paths. Numbers below were measured with the in-tree `uaii_bench` harness.

## How to run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUAII_BUILD_BENCHMARKS=ON
cmake --build build --config Release --target uaii_bench --parallel

# Human-readable
./build/benchmarks/uaii_bench --iters 8

# Machine-readable
./build/benchmarks/uaii_bench --iters 8 --json
```

Windows (Ninja):

```powershell
.\build\benchmarks\uaii_bench.exe --iters 8 --json
```

Also:

```bash
uaii benchmark --demo --iters 50   # planner fusion / memory path
```

## Sample results (ref-tiled CPU GEMM)

**Host:** Windows 11 · MinGW g++ 15 · Release · `GEMM=ref-tiled` (no oneDNN/OpenBLAS)  
**Artifact:** [`benchmarks/results/sample_windows_mingw.json`](../benchmarks/results/sample_windows_mingw.json)

| Workload | Baseline | UAII | Speedup |
|---|---:|---:|---:|
| f32 GEMM **512³** (naive ijk) | 146 ms | **24 ms** | **6.2×** |
| f32 GEMM **1024³** (naive ijk) | 6187 ms | **452 ms** | **13.7×** |
| Session MLP (MatMul→ReLU→MatMul→Softmax) | — | **~0.01 ms**/iter | — |
| Q4_0 MatMul vs unpack+f32 (1×1024 @ 1024×4096) | 12.1 ms | **8.0 ms** | **1.5×** |
| Q4_0 weight footprint vs f32 | 16.0 MiB | **2.25 MiB** | **7.1× smaller** |

Peak GEMM throughput in this run: **~11 GFLOP/s** at 512³ on the tiled ref provider.

> Linking **oneDNN** or **OpenBLAS** (`-DUAII_WITH_ONEDNN=ON` / `UAII_WITH_OPENBLAS=ON`) and CUDA (`UAII_WITH_CUDA=ON`) typically increases absolute GFLOP/s further. Publish those numbers from your machine with `--json` and cite the provider string from `uaii doctor`.

## Methodology

- **Naive GEMM:** classic triple-loop `ijk` MatMul — intentional scalar-ish baseline (no tiling, no threading).
- **UAII GEMM:** `kernels::default_gemm()` (tiled + threaded ref, or vendor BLAS when built).
- **Session:** one warm session, timed `Session::run` only (create excluded).
- **Quant:** synthetic Q4_0 packs; compares full-row unpack + f32 GEMM vs `quant_gemm_f32` packed path.
- Warmup iterations discarded; reported values are averages over `--iters`.

## Fairness notes

These benches measure **UAII’s own kernels and session path**, not a bake-off against llama.cpp / ONNX Runtime / TensorRT. Absolute tokens/s on large LLMs depends on model, quant, CPU/GPU SKU, and memory bandwidth. Use `uaii_bench` + your hardware to produce comparable numbers for PRs and release notes.
