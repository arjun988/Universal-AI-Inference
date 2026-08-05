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
- Download from the run’s **Artifacts** section

## Published results — GitHub Actions Windows

**Host:** GitHub Actions `windows-latest` · AMD EPYC 7763 · **4 threads** · Release · `GEMM=ref-tiled`  
**Run:** https://github.com/arjun988/Universal-AI-Inference/actions/runs/31042515292  
**JSON:** [`benchmarks/results/ci_windows_gha.json`](../benchmarks/results/ci_windows_gha.json)  
(raw artifact copy: [`ci_windows_1f263e31fd60.json`](../benchmarks/results/ci_windows_1f263e31fd60.json))

| Field | Value |
|---|---|
| Statistic | **median** of 21 trials (warmup 5; N=1024 used 10 trials) |
| Scope | Kernel microbenchmarks — not LLM tokens/s |

### 1. Dense f32 GEMM

Square `C = A @ B`, FLOPs = `2·N³`.

| N | Median ms | GFLOP/s |
|---:|---:|---:|
| 256 | 9.96 | **3.37** |
| 512 | 85.2 | **3.15** |
| 1024 | 1173 | **1.83** |

These absolute numbers reflect a **4-thread shared CI runner**, not a tuned desktop. Linking **oneDNN** or **OpenBLAS** and re-running is the fair way to quote higher throughput — always cite `gemm_provider`.

### 2. Session graph

Synthetic **8 × MatMul(512²)+ReLU + Softmax** (`Session::run` only):

| Metric | Value |
|---|---:|
| Parameters (f32) | 2,097,152 |
| Median `Session::run` | **3.87 ms** |

### 3. Q4_0 weights (format + kernel)

| Metric | Value |
|---|---:|
| Weight shape | 2048 × 4096 |
| f32 footprint | 32.0 MiB |
| Q4_0 packed | **4.5 MiB** |
| Format compression | **7.11×** (GGUF Q4_0: 32 values / 18 bytes) |
| Packed quant-GEMM | **5.35 ms** |
| Unpack-all + f32 GEMM | 13.6 ms |

Memory ratio is **format-defined**. Timing compares two UAII paths on a synthetic Q4_0 payload, not a full GGUF model.

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

> UAII `ref-tiled` CPU GEMM on GitHub Actions `windows-latest` (AMD EPYC 7763, 4 threads), median of 21 trials: 3.15 GFLOP/s at 512³. JSON: `benchmarks/results/ci_windows_gha.json`. Not an LLM tokens/s result.
