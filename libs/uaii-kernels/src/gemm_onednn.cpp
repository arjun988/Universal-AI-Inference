#include "uaii/kernels/gemm.hpp"

#if defined(UAII_HAVE_ONEDNN)
#  include <dnnl.hpp>
#endif

namespace uaii {
namespace kernels {

#if defined(UAII_HAVE_ONEDNN)

namespace {

class OneDnnGemm final : public IGemm {
 public:
  GemmProvider provider() const noexcept override { return GemmProvider::OneDnn; }
  const char* name() const noexcept override { return "onednn"; }

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
    try {
      dnnl::engine eng(dnnl::engine::kind::cpu, 0);
      dnnl::stream s(eng);
      const dnnl::memory::dim M = m, N = n, K = k;
      dnnl::memory::dims a_dims = transpose_a ? dnnl::memory::dims{K, M}
                                              : dnnl::memory::dims{M, K};
      dnnl::memory::dims b_dims = transpose_b ? dnnl::memory::dims{N, K}
                                              : dnnl::memory::dims{K, N};
      dnnl::memory::dims c_dims{M, N};
      auto a_md = dnnl::memory::desc(
          a_dims, dnnl::memory::data_type::f32,
          transpose_a ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
      auto b_md = dnnl::memory::desc(
          b_dims, dnnl::memory::data_type::f32,
          transpose_b ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
      auto c_md =
          dnnl::memory::desc(c_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
      dnnl::memory a_mem(a_md, eng, const_cast<float*>(a));
      dnnl::memory b_mem(b_md, eng, const_cast<float*>(b));
      dnnl::memory c_mem(c_md, eng, c);
      (void)lda;
      (void)ldb;
      (void)ldc;
      auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md);
      dnnl::matmul(pd).execute(s, {{DNNL_ARG_SRC, a_mem},
                                   {DNNL_ARG_WEIGHTS, b_mem},
                                   {DNNL_ARG_DST, c_mem}});
      s.wait();
      if (alpha != 1.0f || beta != 0.0f) {
        // oneDNN matmul above assumes alpha=1 beta=0; apply scale if needed.
        if (beta != 0.0f || alpha != 1.0f) {
          for (std::int64_t i = 0; i < m * n; ++i) {
            // When beta!=0 this path is incomplete for general epilogue — fall back.
          }
        }
        if (beta != 0.0f) {
          return Error::make(ErrorCode::NotImplemented,
                             "onednn gemm beta!=0 not supported; use ref");
        }
        if (alpha != 1.0f) {
          for (std::int64_t i = 0; i < m; ++i)
            for (std::int64_t j = 0; j < n; ++j) c[i * ldc + j] *= alpha;
        }
      }
      return Error::success();
    } catch (const std::exception& ex) {
      return Error::make(ErrorCode::Internal, std::string("onednn: ") + ex.what());
    }
  }
};

OneDnnGemm& onednn_gemm() {
  static OneDnnGemm g;
  return g;
}

}  // namespace

IGemm* try_make_onednn() { return &onednn_gemm(); }

#else

IGemm* try_make_onednn() { return nullptr; }

#endif

}  // namespace kernels
}  // namespace uaii
