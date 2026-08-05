# Phase 5 — Hardware Expansion

**Status:** In-tree **scaffold** — GPU backend *names* execute via host-fallback CPU kernels unless a real device path is implemented under `UAII_WITH_*`.

## Delivered (honest)

| Item | Reality |
|---|---|
| Backend factory | `backends::list_backends` / `create_backend` |
| Host executable base | `HostExecutableBackend` (host memory + CPU kernels) |
| CPU backend | Real CPU execution |
| CUDA / Metal / Vulkan / WebGPU / ROCm | **Host-fallback by default**; parity demos are CPU-vs-CPU under GPU names |
| Native scaffolds | Optional `-DUAII_WITH_*=ON` still dispatches to CPU math today |
| Parity policy | `backends::ParityPolicy` + `compare_f32_buffers` |
| Doctor | Probes backends; labels host-fallback honestly |

## Not done

- Real device kernels / memory / streams for CUDA, Metal, Vulkan, WebGPU, ROCm

## Verify locally

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel

uaii doctor
uaii run --demo parity
```
