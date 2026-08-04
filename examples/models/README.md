# Model fixtures (Phase 4)

Demos generate tiny GGUF / Safetensors files here:

```bash
uaii run --demo gguf          # writes tiny_demo.gguf and runs generate-style forward
uaii run --demo safetensors   # writes tiny_demo.safetensors
uaii convert examples/models/tiny_demo.gguf -o examples/models/tiny_demo.uaii.json
```

These are synthetic models for loader/runtime validation, not production checkpoints.
