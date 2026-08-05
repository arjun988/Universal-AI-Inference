# Phase 7 — Ecosystem

**Status:** C API + Python SDK + static docs site ship in-tree. Pre-1.0 ABI; wheels do not auto-ship natives unless you bundle them.

## Delivered

| Item | Reality |
|---|---|
| C API | **0.2.0** (`struct_size`, fail-closed weights) — not 1.0.0 |
| Shared ABI library | `uaii_capi` |
| Python SDK | ctypes (+ optional pybind11); looks for `uaii/_native/` or `UAII_CAPI_PATH` |
| Bundle helper | `bindings/python/scripts/bundle_native.py` copies built `uaii_capi` into the package |
| Docs site | `website/` Next.js static export |
| `find_package(uaii)` | Install export + `uaiiConfig.cmake` (`find_dependency(Threads)`) |
| Marketplace | **Design only** — `docs/marketplace.md` |

## Not done

- Published PyPI wheels with platform binaries by default
- Out-of-process plugin sandbox
- Marketplace service

## Verify locally

```bash
# After building C++:
python bindings/python/scripts/bundle_native.py --build-dir build
pip install -e bindings/python
python examples/python/load_run_profile.py

cd website && npm ci && npm run build
```
