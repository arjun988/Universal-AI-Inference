# Phase 2 — UAII IR (Complete in-tree)

**Status:** Implemented in repository (verify locally via README build steps)

## Delivered

- Graph, tensor, node, attribute, and operator schema definitions (`include/uaii/ir`)
- Dynamic operator registry + built-in schemas (MatMul, Softmax, Attention, …)
- Graph validator (ids, arity, cycles, dtype/shape presence, IR version)
- Serialization:
  - FlatBuffers contract: `schemas/uaii_ir.fbs`
  - Native binary codec (`.uaii`, magic `UAIR`) — zero external deps
  - JSON codec (`.uaii.json` / `.json`) for hand-authored graphs
- Execution plan data structures + topological builder
- IR versioning (`1.0`, compatibility rules)
- CLI: `uaii validate`, `uaii inspect`, `uaii graph`
- Example: `examples/ir/toy_mlp.uaii.json`

## Verify locally

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel
uaii validate examples/ir/toy_mlp.uaii.json
uaii inspect examples/ir/toy_mlp.uaii.json
uaii graph examples/ir/toy_mlp.uaii.json --format plan
uaii graph examples/ir/toy_mlp.uaii.json --format dot
```

Round-trip (optional): load JSON, save `.uaii`, load binary, validate again.
