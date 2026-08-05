#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace uaii {
namespace kernels {

enum class GemmProvider {
  Ref = 0,
  OneDnn,
  OpenBlas,
  CudaCublasLt,
};

[[nodiscard]] UAII_API const char* to_string(GemmProvider p) noexcept;

/// Portable GEMM: C = alpha * op(A) * op(B) + beta * C  (row-major f32).
class UAII_API IGemm {
 public:
  virtual ~IGemm() = default;
  [[nodiscard]] virtual GemmProvider provider() const noexcept = 0;
  [[nodiscard]] virtual const char* name() const noexcept = 0;

  [[nodiscard]] virtual Error gemm_f32(std::int64_t m,
                                       std::int64_t n,
                                       std::int64_t k,
                                       const float* a,
                                       std::int64_t lda,
                                       bool transpose_a,
                                       const float* b,
                                       std::int64_t ldb,
                                       bool transpose_b,
                                       float* c,
                                       std::int64_t ldc,
                                       float alpha = 1.0f,
                                       float beta = 0.0f) = 0;
};

/// Process-wide GEMM registry (Ref always available; oneDNN/OpenBLAS optional).
class UAII_API GemmRegistry {
 public:
  static GemmRegistry& instance();

  void set_preferred(GemmProvider p);
  [[nodiscard]] IGemm& active();
  [[nodiscard]] const IGemm& active() const;
  [[nodiscard]] GemmProvider active_provider() const noexcept;
  [[nodiscard]] std::string describe() const;

 private:
  GemmRegistry();
  GemmProvider preferred_ = GemmProvider::Ref;
};

[[nodiscard]] UAII_API IGemm& default_gemm();

}  // namespace kernels
}  // namespace uaii
