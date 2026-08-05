#include "uaii/kernels/kernels.hpp"
#include "uaii/kernels/gemm.hpp"
#include "uaii/kernels/view_util.hpp"

namespace uaii {
namespace kernels {
namespace {

Error require_f32_2d(const TensorView& t, const char* name) {
  if (t.dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, std::string(name) + " must be f32");
  }
  if (t.rank != 2 || t.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string(name) + " must be rank-2 with data");
  }
  return check_view_bytes(t, name);
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
  err = check_view_bytes(*c, "C");
  if (!err.ok()) return err;

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

  return default_gemm().gemm_f32(a_rows, b_cols, a_cols, a.f32(), a.dim(1), transpose_a,
                                 b.f32(), b.dim(1), transpose_b, c->f32(), c->dim(1));
}

}  // namespace kernels
}  // namespace uaii
