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

Doctor prints GEMM provider and per-backend capability details. Silent “GPU name, CPU math” without `host_fallback` in caps is a bug.
