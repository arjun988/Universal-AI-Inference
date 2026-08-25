#include "cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace uaii {
namespace backends {
namespace cuda_kernels {
namespace {

__global__ void add_kernel(const float* a, const float* b, float* c, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void mul_kernel(const float* a, const float* b, float* c, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] * b[i];
  }
}

__global__ void silu_kernel(const float* x, float* y, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

__global__ void relu_kernel(const float* x, float* y, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    y[i] = v > 0.0f ? v : 0.0f;
  }
}

// Softmax over axis with layout [outer, axis, inner].
__global__ void softmax_kernel(const float* x, float* y, std::int64_t outer,
                               std::int64_t axis, std::int64_t inner) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t rows = outer * inner;
  if (row >= rows) {
    return;
  }
  const std::int64_t o = row / inner;
  const std::int64_t in = row % inner;
  float max_v = -INFINITY;
  for (std::int64_t a = 0; a < axis; ++a) {
    const float v = x[(o * axis + a) * inner + in];
    if (v > max_v) {
      max_v = v;
    }
  }
  float sum = 0.0f;
  for (std::int64_t a = 0; a < axis; ++a) {
    const float e = expf(x[(o * axis + a) * inner + in] - max_v);
    y[(o * axis + a) * inner + in] = e;
    sum += e;
  }
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (std::int64_t a = 0; a < axis; ++a) {
    y[(o * axis + a) * inner + in] *= inv;
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

inline dim3 grid_for(std::size_t n, int block = 256) {
  return dim3(static_cast<unsigned>((n + static_cast<std::size_t>(block) - 1) /
                                    static_cast<std::size_t>(block)));
}

inline cudaStream_t as_stream(void* stream) {
  return static_cast<cudaStream_t>(stream);
}

}  // namespace

int launch_add_f32(const float* a, const float* b, float* c, std::size_t n, void* stream) {
  if (n == 0) {
    return 0;
  }
  add_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(a, b, c, n);
  return static_cast<int>(cudaGetLastError());
}

int launch_mul_f32(const float* a, const float* b, float* c, std::size_t n, void* stream) {
  if (n == 0) {
    return 0;
  }
  mul_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(a, b, c, n);
  return static_cast<int>(cudaGetLastError());
}

int launch_silu_f32(const float* x, float* y, std::size_t n, void* stream) {
  if (n == 0) {
    return 0;
  }
  silu_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, y, n);
  return static_cast<int>(cudaGetLastError());
}

int launch_relu_f32(const float* x, float* y, std::size_t n, void* stream) {
  if (n == 0) {
    return 0;
  }
  relu_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, y, n);
  return static_cast<int>(cudaGetLastError());
}

int launch_softmax_f32(const float* x, float* y, std::int64_t outer, std::int64_t axis,
                       std::int64_t inner, void* stream) {
  const std::size_t rows = static_cast<std::size_t>(outer * inner);
  if (rows == 0 || axis <= 0) {
    return 0;
  }
  softmax_kernel<<<grid_for(rows), 256, 0, as_stream(stream)>>>(x, y, outer, axis, inner);
  return static_cast<int>(cudaGetLastError());
}

int launch_rmsnorm_f32(const float* x, const float* weight, float* y, std::int64_t rows,
                       std::int64_t cols, float eps, void* stream) {
  if (rows <= 0 || cols <= 0) {
    return 0;
  }
  rmsnorm_kernel<<<grid_for(static_cast<std::size_t>(rows)), 256, 0, as_stream(stream)>>>(
      x, weight, y, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}

namespace {

// One block per row; threads cover rotary half-pairs (NeoX) or interleaved pairs.
__global__ void rope_kernel(const float* x, float* y, std::int64_t rows, std::int64_t dim,
                            std::int64_t rope_chunk, std::int64_t rotary_dim, int mode,
                            float theta, float base_pos, const float* positions,
                            const float* inv_freq, float mscale) {
  const std::int64_t r = static_cast<std::int64_t>(blockIdx.x);
  if (r >= rows) {
    return;
  }
  const float pos =
      (positions != nullptr) ? positions[r] : (base_pos + static_cast<float>(r));
  const float* row = x + r * dim;
  float* out_row = y + r * dim;
  const std::int64_t n_chunks = dim / rope_chunk;
  const std::int64_t rot_off = rope_chunk - rotary_dim;
  const std::int64_t half = rotary_dim / 2;
  const bool neox = (mode == 2);
  const float ms = mscale != 0.f ? mscale : 1.f;

  for (std::int64_t h = 0; h < n_chunks; ++h) {
    const float* head = row + h * rope_chunk;
    float* out_head = out_row + h * rope_chunk;
    for (std::int64_t c = threadIdx.x; c < rot_off; c += blockDim.x) {
      out_head[c] = head[c];
    }
    const float* rot_in = head + rot_off;
    float* rot_out = out_head + rot_off;
    if (neox) {
      for (std::int64_t i = threadIdx.x; i < half; i += blockDim.x) {
        const float freq =
            (inv_freq != nullptr)
                ? inv_freq[i]
                : (1.0f / powf(theta, (2.0f * static_cast<float>(i)) /
                                           static_cast<float>(rotary_dim)));
        const float angle = pos * freq;
        const float c = cosf(angle) * ms;
        const float s = sinf(angle) * ms;
        const float a = rot_in[i];
        const float b = rot_in[i + half];
        rot_out[i] = a * c - b * s;
        rot_out[i + half] = a * s + b * c;
      }
    } else {
      for (std::int64_t i = threadIdx.x * 2; i < rotary_dim; i += blockDim.x * 2) {
        const std::int64_t pair = i / 2;
        const float freq =
            (inv_freq != nullptr)
                ? inv_freq[pair]
                : (1.0f / powf(theta, static_cast<float>(i) / static_cast<float>(rotary_dim)));
        const float angle = pos * freq;
        const float c = cosf(angle) * ms;
        const float s = sinf(angle) * ms;
        const float a = rot_in[i];
        const float b = rot_in[i + 1];
        rot_out[i] = a * c - b * s;
        rot_out[i + 1] = a * s + b * c;
      }
    }
  }
}

__global__ void logit_softcap_kernel(const float* x, float* y, std::size_t n, float cap) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = cap * tanhf(x[i] / cap);
  }
}

}  // namespace

int launch_rope_f32(const float* x, float* y, std::int64_t rows, std::int64_t dim,
                    std::int64_t rope_chunk, std::int64_t rotary_dim, int mode, float theta,
                    float base_pos, const float* positions, const float* inv_freq, float mscale,
                    void* stream) {
  if (rows <= 0 || dim <= 0 || rope_chunk <= 0 || rotary_dim <= 0) {
    return 0;
  }
  const int block = 256;
  rope_kernel<<<static_cast<unsigned>(rows), block, 0, as_stream(stream)>>>(
      x, y, rows, dim, rope_chunk, rotary_dim, mode, theta, base_pos, positions, inv_freq,
      mscale);
  return static_cast<int>(cudaGetLastError());
}

int launch_logit_softcap_f32(const float* x, float* y, std::size_t n, float cap,
                             void* stream) {
  if (n == 0 || cap <= 0.0f) {
    return 0;
  }
  logit_softcap_kernel<<<grid_for(n), 256, 0, as_stream(stream)>>>(x, y, n, cap);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace cuda_kernels
}  // namespace backends
}  // namespace uaii
