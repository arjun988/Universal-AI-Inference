# Benchmarks

## Microbench harness (`uaii_bench`)

Builds with `-DUAII_BUILD_BENCHMARKS=ON` (default ON).

```bash
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --iters 8
./build/benchmarks/uaii_bench --iters 8 --json
```

Measures:

1. Naive f32 GEMM vs UAII `IGemm` (512³ / 1024³)
2. Fused session MLP latency
3. Q4_0 packed GEMM vs unpack-then-f32 (+ memory footprint)

Published sample: [`results/sample_windows_mingw.json`](results/sample_windows_mingw.json)  
Write-up: [`docs/benchmarks.md`](../docs/benchmarks.md)

## CLI planner benchmark

```bash
uaii benchmark --demo --iters 50
```

Compares baseline vs fused / memory-reuse session path.

## Python

```bash
python benchmarks/bench_python_session.py --iters 30
```
