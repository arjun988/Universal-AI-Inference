# Benchmarks

UAII publishes **absolute, reproducible microbenchmarks** (`uaii_bench` schema **v3**) — not strawman speedups, and not a bake-off against llama.cpp / ONNX Runtime / TensorRT.

Suites: **gemm** (per linked provider), **bandwidth**, **attention**, **session**, **quant**.

## How to run

```bash
sudo apt install -y libopenblas-dev libdnnl-dev   # optional vendors
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DUAII_BUILD_BENCHMARKS=ON \
  -DUAII_WITH_ONEDNN=ON -DUAII_WITH_OPENBLAS=ON
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --suite all --providers all --trials 21 --json
```

WSL: `TRIALS=21 bash scripts/run_bench_wsl.sh`

## Published results — local WSL

**Host:** WSL2 · Intel Core i9-14900HX · 32 threads · Release  
**JSON:** [`benchmarks/results/local_wsl.json`](../benchmarks/results/local_wsl.json)  
**Linked:** `ref`, `openblas` (oneDNN not installed in this capture)

### GEMM by provider

| Provider | 256³ | 512³ | 1024³ |
|---|---:|---:|---:|
| **openblas** | **141 GFLOP/s** | **284 GFLOP/s** | **425 GFLOP/s** |
| ref-tiled | 7.7 GFLOP/s | 11.8 GFLOP/s | 15.1 GFLOP/s |

OpenBLAS is ~**28×** the ref kernel at 1024³ on this machine — cite both; the ref path is the always-available baseline.

### Bandwidth / attention / session / Q4

| Metric | Value |
|---|---:|
| STREAM triad (~256 MiB) | **16.6 GB/s** |
| Attention e2e (B1 H8 S512 D64, ref GEMM) | **59.7 ms** |
| Session 8×512 stack | **2.77 ms** |
| Q4_0 format compression | **7.11×** |

## Methodology

Median of 21 trials (10 for large kernels), full env in JSON. Not LLM tokens/s.

## CI artifacts (not anecdotal)

Every push/PR runs the `benchmarks` job (Ubuntu / Windows / macOS). It:

1. Builds `uaii_bench` (Linux with OpenBLAS + oneDNN when packages install)
2. Runs `--suite all --providers all --trials 21 --json`
3. Uploads `uaii-bench-<os>-<sha>` artifacts (90-day retention)
4. Writes a job summary table

**How to cite CI:** open the workflow run → Artifacts → download JSON → quote `cpu`, `linked_providers`, and `gflops_median` from `gemm_by_provider`. Always include the commit SHA and runner OS.

**How to rerun locally (same flags as CI):**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DUAII_BUILD_BENCHMARKS=ON -DUAII_WITH_OPENBLAS=ON -DUAII_WITH_ONEDNN=ON
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --suite all --providers all --trials 21 --warmup 5 --json \
  | tee benchmarks/results/local.json
```

WSL helper: `TRIALS=21 bash scripts/run_bench_wsl.sh`

## Fair citation

> UAII with OpenBLAS on Intel Core i9-14900HX (32 threads, WSL2), median of 21: **425 GFLOP/s** at 1024³ f32 GEMM (ref-tiled 15.1 GFLOP/s). JSON: `benchmarks/results/local_wsl.json`.
