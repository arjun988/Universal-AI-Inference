# Phase 3 — CPU Runtime (Complete in-tree)

**Status:** Implemented in repository (verify locally via README build steps)

## Delivered

| Area | Implementation |
|---|---|
| Memory | `uaii-memory`: budget, arena, tensor pool, allocator |
| Scheduler | `CpuScheduler` — CPU placement, plan order |
| Execution engine | `runtime::Session` lifecycle (create / feed / run / fetch) |
| CPU backend | `backends::CpuBackend` |
| Kernels | MatMul, Softmax, LayerNorm, RMSNorm, Relu, Gelu, Silu, Add, Mul, Identity |
| Transformer-style | `uaii run --demo tiny_block` (LN → MatMul → Softmax → MatMul → Add → RMSNorm) |
| Toy E2E | `uaii run --demo toy_mlp` with deterministic expected output |

## Verify locally

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel

uaii run --demo toy_mlp
uaii run --demo tiny_block

uaii run examples/ir/toy_mlp.uaii.json \
  --weight-init ones \
  --input x=1,2,3,4 \
  --output y_prob
```

Expected for `toy_mlp` demo: `y_prob = [0.25, 0.25, 0.25, 0.25]`.
