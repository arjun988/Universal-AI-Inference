#pragma once

#include <cstddef>
#include <cstdint>

namespace uaii {
namespace backends {
namespace rocm_kernels {

[[nodiscard]] int launch_add_f32(const float* a, const float* b, float* c, std::size_t n,
                                 void* stream);
[[nodiscard]] int launch_rmsnorm_f32(const float* x, const float* weight, float* y,
                                     std::int64_t rows, std::int64_t cols, float eps,
                                     void* stream);

}  // namespace rocm_kernels
}  // namespace backends
}  // namespace uaii
