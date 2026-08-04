# Python examples (Phase 7)

```bash
# From repo root — after building uaii_capi
cmake -S . -B build
cmake --build build --config Release --parallel

# Run without pip install
python examples/python/load_run_profile.py
```

Or install the SDK:

```bash
pip install -e bindings/python
python examples/python/load_run_profile.py
```
