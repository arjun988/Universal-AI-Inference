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

}  // namespace cuda_kernels
}  // namespace backends
}  // namespace uaii
