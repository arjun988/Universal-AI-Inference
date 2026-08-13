# Phase 5 — Hardware Expansion

**Status:** Partial native GPU exists (CUDA / Metal / ROCm MatMul and selected elementwise ops when `UAII_WITH_*=ON`); CUDA also runs **RoPE** and **Attention decode (seq=1)** on-device/staged. Metal/ROCm Attention/RoPE still host-fallback. Vulkan/WebGPU remain mostly scaffold.

## Delivered (honest)

| Item | Reality |
|---|---|
| Backend factory | `backends::list_backends` / `create_backend` |
| Host executable base | `HostExecutableBackend` (host memory + CPU kernels) |
| CPU backend | Real CPU execution |
| CUDA (native) | Device memory + cuBLASLt MatMul; Add/Mul/RMSNorm/Softmax/Silu on-device when built; **RoPE** + **Attention decode (seq=1)** staged/on-device; `attention_host_fallback==false` when native init OK; Attention prefill / MLAExpandK → CPU |
| Metal (native) | Shared buffers + MSL MatMul/Add/RMSNorm when pipelines build; Attention/RoPE → CPU |
| ROCm (native) | HIP memory + rocBLAS MatMul; selected HIP kernels; Attention/RoPE → CPU |
| Vulkan / WebGPU | Minimal / experimental; mostly host math; Attention/RoPE → CPU |
| Parity policy | `backends::ParityPolicy` + `compare_f32_buffers` |
| Doctor | Probes backends; labels `host_fallback` / `attention_host_fallback` honestly |

## Not done

- On-device **Attention** (decode) and **RoPE** for CUDA (staged H2D when session tensors are host); Metal / ROCm Attention/RoPE still scheduled to host; CUDA Attention prefill (seq>1) still CPU
- Full Vulkan MatMul compute path / WebGPU compute dispatch
- Claiming “full GPU inference” while Attention runs on CPU

## Verify locally

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel

uaii doctor
uaii run --demo parity
```

See also [backend_support.md](../backend_support.md).
