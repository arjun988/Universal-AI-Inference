# UAII Python SDK

Load a model (or UAII IR), run inference, and profile — without reading C++ internals.

## Prerequisites

1. Build the C++ runtime (produces `uaii_capi` shared library):

```bash
cmake -S ../.. -B ../../build -DUAII_BUILD_TESTS=ON
cmake --build ../../build --config Release --parallel
```

2. Optional native extension (pybind11):

```bash
cmake -S ../.. -B ../../build -DUAII_BUILD_PYTHON=ON
cmake --build ../../build --config Release --parallel
```

3. Install this package editable:

```bash
pip install -e .
# or: PYTHONPATH=bindings/python
```

If the shared library is not on the default search path:

```bash
# Windows PowerShell
$env:UAII_CAPI_PATH = "C:\path\to\uaii_capi.dll"
```

## Quick start

```python
import uaii

print(uaii.version(), "c_api", uaii.c_api_version())

session = uaii.Session.from_path(
    "../../examples/ir/toy_mlp.uaii.json",
    weight_init="ones",
    profile=True,
    trace_path="uaii_py_profile.json",
)
session.set_tensor("x", [1.0, 2.0, 3.0, 4.0])
session.run()
print(session.get_tensor("y_prob"))
print(session.profile_summary())
```

See `examples/python/` in the repo root.
