# Phase 5 — Hardware Expansion (Complete in-tree)

**Status:** Implemented in repository (verify locally via README build steps)

## Delivered

| Item | Implementation |
|---|---|
| Backend factory | `backends::list_backends` / `create_backend` |
| Host executable base | `HostExecutableBackend` (host memory + CPU kernels) |
| CPU backend | `CpuBackend` |
| CUDA / Metal / Vulkan / WebGPU / ROCm | Executable backends with **host-fallback** (no SDK required) |
| Native scaffolds | Optional `-DUAII_WITH_CUDA|METAL|VULKAN|WEBGPU|ROCM=ON` |
| Parity policy | `backends::ParityPolicy` + `compare_f32_buffers` |
| Multi-backend session | `SessionOptions.backend_name` |
| Parity demo / CLI | `uaii run --demo parity`, `--backend`, `--force-host-fallback` |
| Doctor | Lists all backends + native_compiled flags |

## Design notes

- Every GPU backend can execute the same f32 IR graph via host-fallback kernels.
- Numerical parity is evaluated under a documented atol/rtol policy (`ParityPolicy`).
- Optional `UAII_WITH_*` flags enable a **native scaffold** path (same math today; real device kernels can replace `native_*_dispatch` later). Vendor SDKs are **not** required to build or verify Phase 5 exit criteria.

## Verify locally

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel

uaii doctor
uaii run --demo parity

# Same IR on multiple backends (host-fallback)
uaii run examples/ir/toy_mlp.uaii.json --backend cpu --weight-init ones --input x=1,2,3,4 --output y_prob
uaii run examples/ir/toy_mlp.uaii.json --backend cuda --force-host-fallback --weight-init ones --input x=1,2,3,4 --output y_prob
uaii run examples/ir/toy_mlp.uaii.json --backend metal --force-host-fallback --weight-init ones --input x=1,2,3,4 --output y_prob
uaii run examples/ir/toy_mlp.uaii.json --backend vulkan --force-host-fallback --weight-init ones --input x=1,2,3,4 --output y_prob
uaii run examples/ir/toy_mlp.uaii.json --backend webgpu --force-host-fallback --weight-init ones --input x=1,2,3,4 --output y_prob
uaii run examples/ir/toy_mlp.uaii.json --backend rocm --force-host-fallback --weight-init ones --input x=1,2,3,4 --output y_prob
```

Optional native scaffold build (still no vendor install required for the scaffold):

```bash
cmake -S . -B build -DUAII_WITH_CUDA=ON -DUAII_WITH_VULKAN=ON
cmake --build build --config Release --parallel
uaii doctor
```
