# Backend support matrix

| Backend | Default (no SDK) | `UAII_WITH_*=ON` + device present |
|---|---|---|
| CPU | Real kernels + tiled/ref or oneDNN/OpenBLAS GEMM | same |
| CUDA | Host-fallback executable | Real device memory; MatMul via **cuBLASLt** (cuBLAS `sgemm` fallback); Add/Mul/RMSNorm/Softmax/Silu on-device; **Attention** (decode `seq=1`, staged H2D + cuBLAS + softmax) + **RoPE** on-device/staged; `uses_host_fallback()==false` and `attention_host_fallback==false` when native init OK; Attention prefill / MLAExpandK / Embedding still CPU |
| Metal | Host-fallback | Shared `MTLBuffer` alloc/copy; **MatMul/Add/RMSNorm** via runtime-compiled MSL; `uses_host_fallback()==false` when pipelines build; `attention_host_fallback==true` |
| Vulkan | Host-fallback | Host-visible mapped buffers; **Add** via embedded SPIR-V `VkCompute` when pipeline builds; **MatMul/RMSNorm** CPU-on-mapped; `uses_host_fallback()==true` until MatMul is VkCompute; `attention_host_fallback==true` |
| WebGPU | Host-fallback | Adapter/device acquire + MapWrite buffer alloc/copy when headers present; compute dispatch pending; honest caps; `uses_host_fallback()==true` |
| ROCm | Host-fallback | HIP device memory; **MatMul** via rocBLAS; **Add/RMSNorm** via HIP kernels; `uses_host_fallback()==false` when native init OK; `attention_host_fallback==true` |

`DeviceScheduler` places ops on the backend's preferred device. When `attention_host_fallback` is true, **Attention** and **RoPE** are scheduled to CPU. With native CUDA init, they are scheduled to the GPU device and run via staged or on-device kernels (prefill Attention still falls back to CPU inside dispatch).

### Default: CPU, GPU if present

CLI (`generate` / `chat` / `run`) and the dashboard default to `--backend auto`. That resolves to the first **native** GPU that actually has a device (priority: CUDA, Metal, ROCm, Vulkan). If none, it uses CPU. `--backend cpu` (or Settings → CPU) pins the host. A GPU that is installed but built with `UAII_WITH_CUDA=OFF` (host-fallback only) does **not** count as detected — auto stays on CPU.

The C API / Python `Session` still default to `"cpu"` (explicit ABI). Pass `"auto"` there if you want the same probe.

Doctor prints GEMM provider, per-backend capability details, and a `recommended:` line matching auto-select. Silent “GPU name, CPU math” without `host_fallback` in caps is a bug.
