# Benchmarks

## `uaii_bench` (schema v3)

```bash
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --suite all --providers all --trials 21 --json
```

WSL:

```powershell
wsl -e bash scripts/run_bench_wsl.sh
```

Suites: `gemm` · `bandwidth` · `attention` · `session` · `quant`  
Providers (when linked): `ref` · `onednn` · `openblas`

**Published:** [`results/local_wsl.json`](results/local_wsl.json)  
Docs: [`docs/benchmarks.md`](../docs/benchmarks.md)

## CLI planner

```bash
uaii benchmark --demo --iters 50
```

## Python

```bash
python benchmarks/bench_python_session.py --iters 30
```
