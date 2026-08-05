#include "uaii/kernels/gemm.hpp"

#if defined(UAII_HAVE_OPENBLAS)
extern "C" {
void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K, float alpha,
                 const float* A, int lda, const float* B, int ldb, float beta, float* C,
                 int ldc);
#  ifndef CblasRowMajor
#    define CblasRowMajor 101
#    define CblasNoTrans 111
#    define CblasTrans 112
#  endif
}
#endif

namespace uaii {
namespace kernels {

#if defined(UAII_HAVE_OPENBLAS)

namespace {

class OpenBlasGemm final : public IGemm {
 public:
  GemmProvider provider() const noexcept override { return GemmProvider::OpenBlas; }
  const char* name() const noexcept override { return "openblas"; }

  Error gemm_f32(std::int64_t m,
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
                 float alpha,
                 float beta) override {
    cblas_sgemm(CblasRowMajor, transpose_a ? CblasTrans : CblasNoTrans,
                transpose_b ? CblasTrans : CblasNoTrans, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, a, static_cast<int>(lda), b,
                static_cast<int>(ldb), beta, c, static_cast<int>(ldc));
    return Error::success();
  }
};

OpenBlasGemm& openblas_gemm() {
  static OpenBlasGemm g;
  return g;
}

}  // namespace

IGemm* try_make_openblas() { return &openblas_gemm(); }

#else

IGemm* try_make_openblas() { return nullptr; }

#endif

}  // namespace kernels
}  // namespace uaii
