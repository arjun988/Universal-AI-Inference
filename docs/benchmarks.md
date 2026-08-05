# Benchmarks

UAII publishes **absolute, reproducible microbenchmarks** of its own kernels — not strawman speedups, and not a bake-off against llama.cpp, ONNX Runtime, or TensorRT.

Primary metrics: **GFLOP/s** and **median wall time**, with full environment disclosure.

## How to run (publication recipe)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUAII_BUILD_BENCHMARKS=ON
cmake --build build --config Release --target uaii_bench --parallel

# Recommended: odd trial count → clean median
./build/benchmarks/uaii_bench --trials 21 --warmup 5 --json
```

Windows:

```powershell
$env:UAII_BENCH_CPU = "Your Exact CPU Model"   # e.g. AMD Ryzen 9 7950X
.\build\benchmarks\uaii_bench.exe --trials 21 --warmup 5 --json
```

Optional engineering appendix (slow; **not** for marketing headlines):

```bash
./build/benchmarks/uaii_bench --trials 21 --vs-naive --json
```

Also available: `uaii benchmark --demo` (planner fusion / memory-reuse path).

## CI (GitHub Actions)

The `benchmarks` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) builds `uaii_bench` on Ubuntu, Windows, and macOS runners, runs `--trials 21 --warmup 5 --json`, and uploads an artifact:

`uaii-bench-<os>-<sha>` → `benchmarks/results/ci_<os>_<sha12>.json`

- Triggered on push / PR / **workflow_dispatch** (Actions → ci → Run workflow)
- Job summary prints the GEMM table (median ms + GFLOP/s) with CPU and `gemm_provider`
- Use these JSONs for public citation when local Application Control blocks unsigned builds

## Sample results (one Windows host)

**Artifact:** [`benchmarks/results/sample_windows_mingw.json`](../benchmarks/results/sample_windows_mingw.json)

| Field | Value |
|---|---|
| OS / toolchain | Windows 11 · MinGW g++ 15 · Ninja Release |
| GEMM provider | `ref-tiled` (no oneDNN / OpenBLAS) |
| Statistic | mean of timed iters (legacy sample; current harness → **median**) |

### 1. Dense f32 GEMM (headline)

Square `C = A @ B`, FLOPs = `2·N³`.

| N | UAII time | Throughput |
|---:|---:|---:|
| 512 | 23.6 ms | **11.4 GFLOP/s** |
| 1024 | 452 ms | **4.7 GFLOP/s** |

Lower GFLOP/s at 1024 is expected on the ref-tiled path under cache/bandwidth pressure. Linking **oneDNN** or **OpenBLAS** and re-running is the fair way to quote higher absolute throughput — always cite `gemm_provider` from the JSON / `uaii doctor`.

### 2. Q4_0 weights (format + kernel)

| Metric | Value |
|---|---:|
| Weight shape | 1024 × 4096 |
| f32 footprint | 16.0 MiB |
| Q4_0 packed | **2.25 MiB** |
| Format compression | **7.11×** (GGUF Q4_0: 32 values / 18 bytes) |
| Packed quant-GEMM | 8.0 ms |
| Unpack-all + f32 GEMM | 12.1 ms |

Memory ratio is **format-defined**. Timing compares two UAII paths on a synthetic Q4_0 payload (valid block layout), not a full GGUF model.

### 3. Session graph

Current `uaii_bench` times a synthetic **8 × MatMul(512²)+ReLU + Softmax** IR stack (`Session::run` only). Re-run locally for a citeable median; do not use outdated toy-MLP figures.

## Methodology (what reviewers expect)

| Rule | Practice |
|---|---|
| Statistic | Default **median** of `--trials` (21), after `--warmup` (5) |
| Environment | CPU brand (`UAII_BENCH_CPU` or CPUID), thread count, GEMM provider, build type, UAII version |
| FLOPs | `2·M·N·K` for dense GEMM |
| Scope label | Every report states these are kernel microbenchmarks |
| Naive ijk | Optional `--vs-naive` **appendix only** |

## What we do **not** claim

- Tokens/s on public LLMs
- Wins vs llama.cpp / ONNX Runtime / TensorRT / PyTorch
- GPU throughput (unless you build CUDA/Metal/etc. and publish that JSON)
- Chatbot end-to-end latency

## Fair citation template

> UAII `ref-tiled` CPU GEMM on \<CPU\>, Windows Release, median of 21 trials: \<X\> GFLOP/s at 512³. Provider and JSON: \<link\>. Not an LLM tokens/s result.
