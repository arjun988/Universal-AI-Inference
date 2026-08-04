#include "uaii/kernels/kernels.hpp"

namespace uaii {
namespace kernels {
namespace {

Error require_f32_2d(const TensorView& t, const char* name) {
  if (t.dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string(name) + " must be f32");
  }
  if (t.rank != 2 || t.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string(name) + " must be rank-2 with data");
  }
  return Error::ok();
}

}  // namespace

Error matmul_f32(const TensorView& a,
                 const TensorView& b,
                 TensorView* c,
                 bool transpose_a,
                 bool transpose_b) {
  if (c == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "matmul out is null");
  }
  Error err = require_f32_2d(a, "A");
  if (!err.ok()) return err;
  err = require_f32_2d(b, "B");
  if (!err.ok()) return err;
  if (c->dtype != DType::F32 || c->rank != 2 || c->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "C must be rank-2 f32");
  }

  const std::int64_t a_rows = transpose_a ? a.dim(1) : a.dim(0);
  const std::int64_t a_cols = transpose_a ? a.dim(0) : a.dim(1);
  const std::int64_t b_rows = transpose_b ? b.dim(1) : b.dim(0);
  const std::int64_t b_cols = transpose_b ? b.dim(0) : b.dim(1);

  if (a_cols != b_rows) {
    return Error::make(ErrorCode::InvalidArgument,
                       "matmul shape mismatch: inner dims " +
                           std::to_string(a_cols) + " vs " + std::to_string(b_rows));
  }
  if (c->dim(0) != a_rows || c->dim(1) != b_cols) {
    return Error::make(ErrorCode::InvalidArgument, "matmul output shape mismatch");
  }

  const float* A = a.f32();
  const float* B = b.f32();
  float* C = c->f32();
  const std::int64_t M = a_rows;
  const std::int64_t K = a_cols;
  const std::int64_t N = b_cols;

  for (std::int64_t i = 0; i < M; ++i) {
    for (std::int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (std::int64_t k = 0; k < K; ++k) {
        const float av = transpose_a ? A[k * a.dim(1) + i] : A[i * a.dim(1) + k];
        const float bv = transpose_b ? B[j * b.dim(1) + k] : B[k * b.dim(1) + j];
        sum += av * bv;
      }
      C[i * N + j] = sum;
    }
  }
  return Error::ok();
}

}  // namespace kernels
}  // namespace uaii
