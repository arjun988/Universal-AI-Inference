#include "uaii/kernels/kernels.hpp"
#include "uaii/kernels/thread_pool.hpp"
#include "uaii/kernels/view_util.hpp"

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

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

inline float load_a(const float* A, std::int64_t i, std::int64_t k, std::int64_t ld,
                    bool transpose_a) {
  return transpose_a ? A[k * ld + i] : A[i * ld + k];
}

inline float load_b(const float* B, std::int64_t k, std::int64_t j, std::int64_t ld,
                    bool transpose_b) {
  return transpose_b ? B[j * ld + k] : B[k * ld + j];
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

  const float* A = a.f32();
  const float* B = b.f32();
  float* C = c->f32();
  const std::int64_t M = a_rows;
  const std::int64_t K = a_cols;
  const std::int64_t N = b_cols;
  const std::int64_t a_ld = a.dim(1);
  const std::int64_t b_ld = b.dim(1);

  // Parallelize over rows; inner loop uses AVX2 when available and layout is contiguous.
  parallel_for(static_cast<std::size_t>(M), [&](std::size_t ii) {
    const std::int64_t i = static_cast<std::int64_t>(ii);
#if defined(__AVX2__)
    if (!transpose_a && !transpose_b) {
      for (std::int64_t j = 0; j < N; ++j) {
        __m256 acc = _mm256_setzero_ps();
        std::int64_t k = 0;
        for (; k + 8 <= K; k += 8) {
          __m256 av = _mm256_loadu_ps(A + i * a_ld + k);
          __m256 bv = _mm256_set_ps(
              B[(k + 7) * b_ld + j], B[(k + 6) * b_ld + j], B[(k + 5) * b_ld + j],
              B[(k + 4) * b_ld + j], B[(k + 3) * b_ld + j], B[(k + 2) * b_ld + j],
              B[(k + 1) * b_ld + j], B[(k + 0) * b_ld + j]);
          acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; k < K; ++k) sum += A[i * a_ld + k] * B[k * b_ld + j];
        C[i * N + j] = sum;
      }
      return;
    }
#endif
    for (std::int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (std::int64_t k = 0; k < K; ++k) {
        sum += load_a(A, i, k, a_ld, transpose_a) * load_b(B, k, j, b_ld, transpose_b);
      }
      C[i * N + j] = sum;
    }
  });
  return Error::success();
}

}  // namespace kernels
}  // namespace uaii
