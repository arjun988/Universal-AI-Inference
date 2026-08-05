#include "uaii/kernels/kernels.hpp"
#include "uaii/kernels/view_util.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace uaii {
namespace kernels {

Error softmax_f32(const TensorView& in, TensorView* out, int axis) {
  if (out == nullptr || in.data == nullptr || out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "softmax null tensor");
  }
  if (in.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "softmax requires f32");
  }
  {
    Error err = check_view_bytes(in, "softmax_in");
    if (!err.ok()) return err;
    err = check_view_bytes(*out, "softmax_out");
    if (!err.ok()) return err;
  }
  if (in.rank == 0 || in.rank != out->rank) {
    return Error::make(ErrorCode::InvalidArgument, "softmax rank mismatch");
  }
  for (std::size_t i = 0; i < in.rank; ++i) {
    if (in.dim(i) != out->dim(i)) {
      return Error::make(ErrorCode::InvalidArgument, "softmax shape mismatch");
    }
  }

  int ax = axis;
  if (ax < 0) {
    ax += static_cast<int>(in.rank);
  }
  if (ax < 0 || ax >= static_cast<int>(in.rank)) {
    return Error::make(ErrorCode::InvalidArgument, "softmax axis out of range");
  }

  // Softmax over `ax`, treating the tensor as [outer, axis, inner].
  std::size_t outer = 1;
  for (int i = 0; i < ax; ++i) {
    outer *= static_cast<std::size_t>(in.dim(static_cast<std::size_t>(i)));
  }
  const std::size_t axis_n = static_cast<std::size_t>(in.dim(static_cast<std::size_t>(ax)));
  std::size_t inner = 1;
  for (std::size_t i = static_cast<std::size_t>(ax) + 1; i < in.rank; ++i) {
    inner *= static_cast<std::size_t>(in.dim(i));
  }

  const float* x = in.f32();
  float* y = out->f32();

  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t i = 0; i < inner; ++i) {
      float max_v = -std::numeric_limits<float>::infinity();
      for (std::size_t a = 0; a < axis_n; ++a) {
        const std::size_t idx = (o * axis_n + a) * inner + i;
        max_v = std::max(max_v, x[idx]);
      }
      float sum = 0.0f;
      for (std::size_t a = 0; a < axis_n; ++a) {
        const std::size_t idx = (o * axis_n + a) * inner + i;
        const float e = std::exp(x[idx] - max_v);
        y[idx] = e;
        sum += e;
      }
      const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
      for (std::size_t a = 0; a < axis_n; ++a) {
        const std::size_t idx = (o * axis_n + a) * inner + i;
        y[idx] *= inv;
      }
    }
  }
  return Error::success();
}

}  // namespace kernels
}  // namespace uaii
