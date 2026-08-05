# Benchmarks

## Microbench harness (`uaii_bench`)

Builds with `-DUAII_BUILD_BENCHMARKS=ON`.

```bash
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --trials 21 --warmup 5 --json
```

Windows:

```powershell
$env:UAII_BENCH_CPU = "Your Exact CPU Model"
.\build\benchmarks\uaii_bench.exe --trials 21 --warmup 5 --json
```

Reports (absolute metrics first):

1. Dense f32 GEMM — median ms + GFLOP/s at 256³ / 512³ / 1024³  
2. Synthetic session stack — `Session::run` only  
3. Q4_0 packed MatMul — format memory ratio + packed vs unpack+f32 times  

Optional: `--vs-naive` appendix (engineering only; do not lead product claims with it).

Published sample (GitHub Actions Windows): [`results/ci_windows_gha.json`](results/ci_windows_gha.json)  
CI also uploads per-run `uaii-bench-<os>-<sha>` artifacts (see [`docs/benchmarks.md`](../docs/benchmarks.md)).

## CLI planner benchmark

```bash
uaii benchmark --demo --iters 50
```

## Python

```bash
python benchmarks/bench_python_session.py --iters 30
```
