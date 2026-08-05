# Backend support matrix

| Backend | Default (no SDK) | `UAII_WITH_*=ON` + device present |
|---|---|---|
| CPU | Real kernels + tiled/ref or oneDNN/OpenBLAS GEMM | same |
| CUDA | Host-fallback executable | Real device memory; MatMul via **cuBLASLt** (cuBLAS `sgemm` fallback); Add/Mul/RMSNorm/Softmax/Silu on-device; `uses_host_fallback()==false` when native init OK; `attention_host_fallback==true` |
| Metal | Host-fallback | Shared `MTLBuffer` alloc/copy; **MatMul/Add/RMSNorm** via runtime-compiled MSL; `uses_host_fallback()==false` when pipelines build; `attention_host_fallback==true` |
| Vulkan | Host-fallback | Host-visible mapped buffers; **Add** via embedded SPIR-V `VkCompute` when pipeline builds; **MatMul/RMSNorm** CPU-on-mapped; `uses_host_fallback()==true` until MatMul is VkCompute; `attention_host_fallback==true` |
| WebGPU | Host-fallback | Adapter/device acquire + MapWrite buffer alloc/copy when headers present; compute dispatch pending; honest caps; `uses_host_fallback()==true` |
| ROCm | Host-fallback | HIP device memory; **MatMul** via rocBLAS; **Add/RMSNorm** via HIP kernels; `uses_host_fallback()==false` when native init OK; `attention_host_fallback==true` |

`DeviceScheduler` places ops on the backend's preferred device. When `attention_host_fallback` is true (all GPU backends today), **Attention** and **RoPE** are scheduled to CPU. The session run loop honors those decisions: CPU-scheduled ops use the backend's host staging path on native GPU backends.

Doctor prints GEMM provider and per-backend capability details. Silent “GPU name, CPU math” without `host_fallback` in caps is a bug.
