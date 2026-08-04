#include "uaii/kernels/kernels.hpp"

#include <cmath>

namespace uaii {
namespace kernels {
namespace {

Error check_same_f32(const TensorView& a, const TensorView& b, const TensorView* out) {
  if (out == nullptr || a.data == nullptr || b.data == nullptr || out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "elementwise null tensor");
  }
  if (a.dtype != DType::F32 || b.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "elementwise requires f32");
  }
  if (a.numel() != b.numel() || a.numel() != out->numel()) {
    return Error::make(ErrorCode::InvalidArgument, "elementwise numel mismatch");
  }
  return Error::ok();
}

Error check_unary_f32(const TensorView& in, const TensorView* out) {
  if (out == nullptr || in.data == nullptr || out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "unary null tensor");
  }
  if (in.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "unary requires f32");
  }
  if (in.numel() != out->numel()) {
    return Error::make(ErrorCode::InvalidArgument, "unary numel mismatch");
  }
  return Error::ok();
}

}  // namespace

Error relu_f32(const TensorView& in, TensorView* out) {
  Error err = check_unary_f32(in, out);
  if (!err.ok()) return err;
  const float* x = in.f32();
  float* y = out->f32();
  const std::size_t n = in.numel();
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = x[i] > 0.0f ? x[i] : 0.0f;
  }
  return Error::ok();
}

Error gelu_f32(const TensorView& in, TensorView* out) {
  Error err = check_unary_f32(in, out);
  if (!err.ok()) return err;
  // tanh approximation
  constexpr float k0 = 0.7978845608f;  // sqrt(2/pi)
  constexpr float k1 = 0.044715f;
  const float* x = in.f32();
  float* y = out->f32();
  const std::size_t n = in.numel();
  for (std::size_t i = 0; i < n; ++i) {
    const float v = x[i];
    const float u = k0 * (v + k1 * v * v * v);
    y[i] = 0.5f * v * (1.0f + std::tanh(u));
  }
  return Error::ok();
}

Error silu_f32(const TensorView& in, TensorView* out) {
  Error err = check_unary_f32(in, out);
  if (!err.ok()) return err;
  const float* x = in.f32();
  float* y = out->f32();
  const std::size_t n = in.numel();
  for (std::size_t i = 0; i < n; ++i) {
    const float v = x[i];
    y[i] = v / (1.0f + std::exp(-v));
  }
  return Error::ok();
}

Error add_f32(const TensorView& a, const TensorView& b, TensorView* out) {
  Error err = check_same_f32(a, b, out);
  if (!err.ok()) return err;
  const float* x = a.f32();
  const float* y = b.f32();
  float* z = out->f32();
  const std::size_t n = a.numel();
  for (std::size_t i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
  return Error::ok();
}

Error mul_f32(const TensorView& a, const TensorView& b, TensorView* out) {
  Error err = check_same_f32(a, b, out);
  if (!err.ok()) return err;
  const float* x = a.f32();
  const float* y = b.f32();
  float* z = out->f32();
  const std::size_t n = a.numel();
  for (std::size_t i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
  return Error::ok();
}

Error identity_f32(const TensorView& in, TensorView* out) {
  Error err = check_unary_f32(in, out);
  if (!err.ok()) return err;
  if (in.data == out->data) {
    return Error::ok();
  }
  const float* x = in.f32();
  float* y = out->f32();
  const std::size_t n = in.numel();
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = x[i];
  }
  return Error::ok();
}

}  // namespace kernels
}  // namespace uaii
