# UAII Python SDK

Load a model (or UAII IR), run inference, and profile — without reading C++ internals.

## Prerequisites

1. Build the C++ runtime (produces `uaii_capi` shared library):

```bash
cmake -S ../.. -B ../../build -DUAII_BUILD_TESTS=ON
cmake --build ../../build --config Release --parallel
```

2. Bundle the native library into the package (recommended for `pip install`):

```bash
python scripts/bundle_native.py --build-dir ../../build
```

3. Optional native extension (pybind11):

```bash
cmake -S ../.. -B ../../build -DUAII_BUILD_PYTHON=ON
cmake --build ../../build --config Release --parallel
```

4. Install this package editable:

```bash
pip install -e .
```

If the shared library is not found under `uaii/_native/` or the build tree:

```bash
# Windows PowerShell
$env:UAII_CAPI_PATH = "C:\path\to\uaii_capi.dll"
```

Defaults are **fail-closed** (`weight_init="none"`). GPU backend names use host-fallback unless real device kernels exist.

## Build a wheel (with natives)

From the **repo root**, after a CMake build of `uaii_capi`:

```bash
# Linux / macOS
bash bindings/python/scripts/build_wheel.sh

# Windows PowerShell
powershell -File bindings/python/scripts/build_wheel.ps1
```

These scripts:

1. Configure and build with CMake (`-DUAII_BUILD_PLUGINS=ON`; tip: add `-DUAII_BUILD_PYTHON=ON` for pybind).
2. Run `scripts/bundle_native.py` to copy `uaii_capi` into `uaii/_native/`.
3. Run `python -m build` to produce wheels under `bindings/python/dist/`.

Manual equivalent:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
python bindings/python/scripts/bundle_native.py --build-dir build
python -m pip install build
python -m build bindings/python
```

CI builds a CPU wheel on Ubuntu via `.github/workflows/python-wheel.yml` (and the multi-OS `wheels.yml` on tags).

## Quick start

```python
import uaii

print(uaii.version(), "c_api", uaii.c_api_version())

s = uaii.Session()
s.load("path/to/model.uaii.json", backend="cpu", weight_init="ones")
s.set("x", [1.0, 2.0, 3.0, 4.0])
s.run()
print(s.get("y_prob"))
```

See `examples/python/load_run_profile.py` in the repo root.
