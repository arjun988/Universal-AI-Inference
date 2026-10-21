#pragma once

// CUDA elementwise / norm kernels (implemented in cuda_kernels.cu).

#include <cstddef>
#include <cstdint>

namespace uaii {
namespace backends {
namespace cuda_kernels {

[[nodiscard]] int launch_add_f32(const float* a, const float* b, float* c, std::size_t n,
                                 void* stream);
[[nodiscard]] int launch_mul_f32(const float* a, const float* b, float* c, std::size_t n,
                                 void* stream);
[[nodiscard]] int launch_silu_f32(const float* x, float* y, std::size_t n, void* stream);
[[nodiscard]] int launch_relu_f32(const float* x, float* y, std::size_t n, void* stream);
[[nodiscard]] int launch_softmax_f32(const float* x, float* y, std::int64_t outer,
                                     std::int64_t axis, std::int64_t inner, void* stream);
[[nodiscard]] int launch_rmsnorm_f32(const float* x, const float* weight, float* y,
                                     std::int64_t rows, std::int64_t cols, float eps,
                                     void* stream);

/// RoPE: layout [rows, dim]. mode 0 = interleaved pairs; mode 2 = NeoX (i, i+half).
/// positions may be null (use base_pos + row). rotary_dim 0 ⇒ rotate full rope_chunk.
/// inv_freq is [rotary_dim/2] scaled inverse frequencies (host-computed). mscale is YaRN.
/// leading nonzero rotates the first rotary_dim elements (HF partial rotary).
[[nodiscard]] int launch_rope_f32(const float* x, float* y, std::int64_t rows,
                                  std::int64_t dim, std::int64_t rope_chunk,
                                  std::int64_t rotary_dim, int mode, float theta,
                                  float base_pos, const float* positions,
                                  const float* inv_freq, float mscale, int leading,
                                  void* stream);

/// Optional Gemma-2 style softcap: y[i] = cap * tanh(x[i] / cap). In-place OK.
[[nodiscard]] int launch_logit_softcap_f32(const float* x, float* y, std::size_t n,
                                           float cap, void* stream);

}  // namespace cuda_kernels
}  // namespace backends
}  // namespace uaii
