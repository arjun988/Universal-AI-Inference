#include "uaii/kernels/quant_gemm.hpp"
#include "uaii/kernels/thread_pool.hpp"
#include "uaii/kernels/view_util.hpp"
#include "uaii/quant/gguf_dequant.hpp"

#include <vector>

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace uaii {
namespace kernels {
namespace {

std::size_t row_stride_bytes(quant::QuantFormat f, std::int64_t cols) {
  return quant::packed_nbytes(f, static_cast<std::size_t>(cols));
}

float dot_f32(const float* a, const float* b, std::int64_t n) {
  std::int64_t k = 0;
  float sum = 0.f;
#if defined(__ARM_NEON)
  float32x4_t acc = vdupq_n_f32(0.f);
  for (; k + 4 <= n; k += 4) {
    acc = vmlaq_f32(acc, vld1q_f32(a + k), vld1q_f32(b + k));
  }
#  if defined(__aarch64__)
  sum = vaddvq_f32(acc);
#  else
  float tmp[4];
  vst1q_f32(tmp, acc);
  sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#  endif
#endif
  for (; k < n; ++k) sum += a[k] * b[k];
  return sum;
}

}  // namespace

bool supports_quant_gemm(quant::QuantFormat f) noexcept {
  return quant::supports_gguf_dequant(f);
}

Error quant_gemm_f32(const TensorView& a, const TensorView& b_quant, TensorView* c,
                     bool transpose_b) {
  if (c == nullptr || !transpose_b) {
    return Error::make(ErrorCode::InvalidArgument,
                       "quant_gemm expects transpose_b=true (B is [N,K] packed)");
  }
  if (a.dtype != DType::F32 || a.rank != 2 || a.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "quant_gemm A must be f32 [M,K]");
  }
  if (!supports_quant_gemm(b_quant.quant_format) || b_quant.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "quant_gemm B quant unsupported");
  }
  if (c->dtype != DType::F32 || c->rank != 2 || c->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "quant_gemm C must be f32");
  }
  Error err = check_view_bytes(a, "quant_A");
  if (!err.ok()) return err;
  err = check_view_bytes(*c, "quant_C");
  if (!err.ok()) return err;

  const std::int64_t M = a.dim(0);
  const std::int64_t K = a.dim(1);
  const std::int64_t N = b_quant.quant_rows > 0 ? b_quant.quant_rows : b_quant.dim(0);
  const std::int64_t Bk = b_quant.quant_cols > 0 ? b_quant.quant_cols : b_quant.dim(1);
  if (K != Bk || c->dim(0) != M || c->dim(1) != N) {
    return Error::make(ErrorCode::InvalidArgument, "quant_gemm shape mismatch");
  }

  const auto* packed = static_cast<const std::uint8_t*>(b_quant.data);
  const std::size_t stride = row_stride_bytes(b_quant.quant_format, K);
  const float* A = a.f32();
  float* C = c->f32();

  parallel_for(static_cast<std::size_t>(N), [&](std::size_t jj) {
    const std::int64_t j = static_cast<std::int64_t>(jj);
    std::vector<float> row(static_cast<std::size_t>(K));
    Error derr = quant::dequant_gguf_row(b_quant.quant_format,
                                         packed + static_cast<std::size_t>(j) * stride, K,
                                         row.data());
    if (!derr.ok()) return;
    for (std::int64_t i = 0; i < M; ++i) {
      C[i * N + j] = dot_f32(A + i * K, row.data(), K);
    }
  });
  return Error::success();
}

}  // namespace kernels
}  // namespace uaii
