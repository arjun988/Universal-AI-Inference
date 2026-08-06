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

Windows (native `.exe` may be blocked by Application Control / WDAC):

```powershell
$env:UAII_BENCH_CPU = "Your Exact CPU Model"
.\build\benchmarks\uaii_bench.exe --trials 21 --warmup 5 --json
```

**Recommended on locked-down Windows:** run inside WSL:

```powershell
wsl -e bash scripts/run_bench_wsl.sh
```

Writes `benchmarks/results/local_wsl.json`.

Optional engineering appendix (slow; **not** for marketing headlines):

```bash
./build/benchmarks/uaii_bench --trials 21 --vs-naive --json
```

Also available: `uaii benchmark --demo` (planner fusion / memory-reuse path).

## Published results — local WSL

**Host:** WSL2 Ubuntu · Intel Core i9-14900HX · **32 threads** · Release · `GEMM=ref-tiled`  
**JSON:** [`benchmarks/results/local_wsl.json`](../benchmarks/results/local_wsl.json)

| Field | Value |
|---|---|
| Statistic | **median** of 21 trials (warmup 5; N=1024 used 10 trials) |
| Scope | Kernel microbenchmarks — not LLM tokens/s |

### 1. Dense f32 GEMM

Square `C = A @ B`, FLOPs = `2·N³`.

| N | Median ms | GFLOP/s |
|---:|---:|---:|
| 256 | 4.32 | **7.78** |
| 512 | 24.3 | **11.1** |
| 1024 | 147 | **14.6** |

Linking **oneDNN** or **OpenBLAS** and re-running is the fair way to quote higher absolute throughput — always cite `gemm_provider`.

### 2. Session graph

Synthetic **8 × MatMul(512²)+ReLU + Softmax** (`Session::run` only):

| Metric | Value |
|---|---:|
| Parameters (f32) | 2,097,152 |
| Median `Session::run` | **2.86 ms** |

### 3. Q4_0 weights (format + kernel)

| Metric | Value |
|---|---:|
| Weight shape | 2048 × 4096 |
| f32 footprint | 32.0 MiB |
| Q4_0 packed | **4.5 MiB** |
| Format compression | **7.11×** (GGUF Q4_0: 32 values / 18 bytes) |
| Packed quant-GEMM | **3.45 ms** |
| Unpack-all + f32 GEMM | 12.9 ms |

Memory ratio is **format-defined**. Timing compares two UAII paths on a synthetic Q4_0 payload, not a full GGUF model.

## CI (GitHub Actions)

The `benchmarks` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) builds `uaii_bench` on Ubuntu, Windows, and macOS and uploads JSON artifacts for regression visibility. Those shared-runner numbers are **not** the published table above.

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

> UAII `ref-tiled` CPU GEMM on Intel Core i9-14900HX (32 threads, WSL2 Release), median of 21 trials: 11.1 GFLOP/s at 512³ / 14.6 GFLOP/s at 1024³. JSON: `benchmarks/results/local_wsl.json`. Not an LLM tokens/s result.
