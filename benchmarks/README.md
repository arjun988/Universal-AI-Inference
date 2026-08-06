# Benchmarks

## Microbench harness (`uaii_bench`)

Builds with `-DUAII_BUILD_BENCHMARKS=ON`.

```bash
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --trials 21 --warmup 5 --json
```

Windows (if Application Control blocks the `.exe`, use WSL):

```powershell
wsl -e bash scripts/run_bench_wsl.sh
```

Reports (absolute metrics first):

1. Dense f32 GEMM — median ms + GFLOP/s at 256³ / 512³ / 1024³  
2. Synthetic session stack — `Session::run` only  
3. Q4_0 packed MatMul — format memory ratio + packed vs unpack+f32 times  

Optional: `--vs-naive` appendix (engineering only; do not lead product claims with it).

**Published sample:** [`results/local_wsl.json`](results/local_wsl.json)  
Write-up: [`docs/benchmarks.md`](../docs/benchmarks.md)

## CLI planner benchmark

```bash
uaii benchmark --demo --iters 50
```

## Python

```bash
python benchmarks/bench_python_session.py --iters 30
```
