# Benchmarks

## CLI (C++)

```bash
uaii benchmark --demo --iters 50
```

Compares Phase 3-style baseline vs Phase 6 optimized path (fusion + memory reuse).

## Python harness

```bash
python benchmarks/bench_python_session.py --iters 30
```

Requires a built `uaii_capi` (or `-DUAII_BUILD_PYTHON=ON`) and the `examples/ir/toy_mlp.uaii.json` fixture.
