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
