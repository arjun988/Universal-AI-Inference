# Phase 7 — Ecosystem (Complete in-tree)

**Status:** Implemented in repository (verify locally; agent does not install npm/pip/SDKs)

## Delivered

| Item | Location |
|---|---|
| Stable C API (semver 1.0.0) | `include/uaii/c_api/uaii.h`, `version.h`, `docs/c_api_stability.md` |
| Shared ABI library | `libs/uaii-capi` → `uaii_capi` |
| Python SDK | `bindings/python/uaii` (ctypes + optional pybind11 `_uaii`) |
| pybind11 module | `bindings/python` when `-DUAII_BUILD_PYTHON=ON` |
| Docs site (Next.js static) | `website/` (`output: "export"`, no backend) |
| Examples | `examples/python/load_run_profile.py` |
| Benchmarks | `benchmarks/` + existing `uaii benchmark` |
| Marketplace design notes | `docs/marketplace.md` + site page |

## Exit criteria

External developer can load a model/IR, run inference, and profile from Python without reading C++ internals — see `examples/python/load_run_profile.py` and the Python docs page.

## Verify locally

```bash
# C++ + C API
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel
uaii doctor

# Python (ctypes over uaii_capi)
pip install -e bindings/python
python examples/python/load_run_profile.py
python benchmarks/bench_python_session.py --iters 20

# Optional pybind11
cmake -S . -B build -DUAII_BUILD_PYTHON=ON
cmake --build build --config Release --parallel

# Docs site (static)
cd website
npm install
npm run build   # emits website/out
```
