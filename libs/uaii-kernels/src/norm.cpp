#include "uaii/kernels/kernels.hpp"

#include <cmath>

namespace uaii {
namespace kernels {
namespace {

Error check_last_axis_norm(const TensorView& in, const TensorView* out) {
  if (out == nullptr || in.data == nullptr || out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "norm null tensor");
  }
  if (in.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "norm requires f32");
  }
  if (in.rank == 0 || in.rank != out->rank) {
    return Error::make(ErrorCode::InvalidArgument, "norm rank mismatch");
  }
  for (std::size_t i = 0; i < in.rank; ++i) {
    if (in.dim(i) != out->dim(i)) {
      return Error::make(ErrorCode::InvalidArgument, "norm shape mismatch");
    }
  }
  return Error::ok();
}

}  // namespace

Error layernorm_f32(const TensorView& in,
                    const TensorView* weight,
                    const TensorView* bias,
                    TensorView* out,
                    float eps) {
  Error err = check_last_axis_norm(in, out);
  if (!err.ok()) {
    return err;
  }

  const std::size_t axis = static_cast<std::size_t>(in.dim(in.rank - 1));
  std::size_t rows = 1;
  for (std::size_t i = 0; i + 1 < in.rank; ++i) {
    rows *= static_cast<std::size_t>(in.dim(i));
  }

  if (weight != nullptr) {
    if (weight->dtype != DType::F32 || weight->numel() != axis) {
      return Error::make(ErrorCode::InvalidArgument, "LayerNorm weight shape");
    }
  }
  if (bias != nullptr) {
    if (bias->dtype != DType::F32 || bias->numel() != axis) {
      return Error::make(ErrorCode::InvalidArgument, "LayerNorm bias shape");
    }
  }

  const float* x = in.f32();
  float* y = out->f32();
  const float* w = weight ? weight->f32() : nullptr;
  const float* b = bias ? bias->f32() : nullptr;

  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = x + r * axis;
    float* out_row = y + r * axis;
    float mean = 0.0f;
    for (std::size_t i = 0; i < axis; ++i) {
      mean += row[i];
    }
    mean /= static_cast<float>(axis);
    float var = 0.0f;
    for (std::size_t i = 0; i < axis; ++i) {
      const float d = row[i] - mean;
      var += d * d;
    }
    var /= static_cast<float>(axis);
    const float inv = 1.0f / std::sqrt(var + eps);
    for (std::size_t i = 0; i < axis; ++i) {
      float v = (row[i] - mean) * inv;
      if (w) v *= w[i];
      if (b) v += b[i];
      out_row[i] = v;
    }
  }
  return Error::ok();
}

Error rmsnorm_f32(const TensorView& in,
                  const TensorView* weight,
                  TensorView* out,
                  float eps) {
  Error err = check_last_axis_norm(in, out);
  if (!err.ok()) {
    return err;
  }

  const std::size_t axis = static_cast<std::size_t>(in.dim(in.rank - 1));
  std::size_t rows = 1;
  for (std::size_t i = 0; i + 1 < in.rank; ++i) {
    rows *= static_cast<std::size_t>(in.dim(i));
  }

  if (weight != nullptr) {
    if (weight->dtype != DType::F32 || weight->numel() != axis) {
      return Error::make(ErrorCode::InvalidArgument, "RMSNorm weight shape");
    }
  }

  const float* x = in.f32();
  float* y = out->f32();
  const float* w = weight ? weight->f32() : nullptr;

  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = x + r * axis;
    float* out_row = y + r * axis;
    float ms = 0.0f;
    for (std::size_t i = 0; i < axis; ++i) {
      ms += row[i] * row[i];
    }
    ms /= static_cast<float>(axis);
    const float inv = 1.0f / std::sqrt(ms + eps);
    for (std::size_t i = 0; i < axis; ++i) {
      float v = row[i] * inv;
      if (w) v *= w[i];
      out_row[i] = v;
    }
  }
  return Error::ok();
}

}  // namespace kernels
}  // namespace uaii
