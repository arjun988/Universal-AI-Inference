#include "rocm_kernels.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace uaii {
namespace backends {
namespace rocm_kernels {
namespace {

__global__ void add_kernel(const float* a, const float* b, float* c, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y,
                               std::int64_t rows, std::int64_t cols, float eps) {
  const std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const float* row = x + r * cols;
  float acc = 0.0f;
  for (std::int64_t c = 0; c < cols; ++c) {
    const float v = row[c];
    acc += v * v;
  }
  const float inv_rms = rsqrtf(acc / static_cast<float>(cols) + eps);
  float* out = y + r * cols;
  if (weight != nullptr) {
    for (std::int64_t c = 0; c < cols; ++c) {
      out[c] = row[c] * inv_rms * weight[c];
    }
  } else {
    for (std::int64_t c = 0; c < cols; ++c) {
      out[c] = row[c] * inv_rms;
    }
  }
}

inline unsigned grid_for(std::size_t n, int block = 256) {
  return static_cast<unsigned>((n + static_cast<std::size_t>(block) - 1) /
                               static_cast<std::size_t>(block));
}

inline hipStream_t as_stream(void* stream) {
  return static_cast<hipStream_t>(stream);
}

}  // namespace

int launch_add_f32(const float* a, const float* b, float* c, std::size_t n, void* stream) {
  if (n == 0) {
    return 0;
  }
  add_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(a, b, c, n);
  return static_cast<int>(hipGetLastError());
}

int launch_rmsnorm_f32(const float* x, const float* weight, float* y, std::int64_t rows,
                       std::int64_t cols, float eps, void* stream) {
  if (rows <= 0 || cols <= 0) {
    return 0;
  }
  rmsnorm_kernel<<<grid_for(static_cast<std::size_t>(rows)), 256, 0, as_stream(stream)>>>(
      x, weight, y, rows, cols, eps);
  return static_cast<int>(hipGetLastError());
}

}  // namespace rocm_kernels
}  // namespace backends
}  // namespace uaii
