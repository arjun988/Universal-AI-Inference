#include "uaii/kernels/gemm.hpp"
#include "uaii/kernels/thread_pool.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace uaii {
namespace kernels {
namespace {

constexpr std::int64_t kTile = 64;

class RefGemm final : public IGemm {
 public:
  GemmProvider provider() const noexcept override { return GemmProvider::Ref; }
  const char* name() const noexcept override { return "ref-tiled"; }

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
    if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0) {
      return Error::make(ErrorCode::InvalidArgument, "ref gemm args");
    }

    auto load_a = [&](std::int64_t i, std::int64_t kk) -> float {
      return transpose_a ? a[kk * lda + i] : a[i * lda + kk];
    };
    auto load_b = [&](std::int64_t kk, std::int64_t j) -> float {
      return transpose_b ? b[j * ldb + kk] : b[kk * ldb + j];
    };

    if (beta == 0.0f) {
      for (std::int64_t i = 0; i < m; ++i) {
        std::memset(c + i * ldc, 0, static_cast<std::size_t>(n) * sizeof(float));
      }
    } else if (beta != 1.0f) {
      for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) c[i * ldc + j] *= beta;
      }
    }

    // Tiled parallel GEMM (competitive without vendor BLAS).
    const std::int64_t tiles_m = (m + kTile - 1) / kTile;
    parallel_for(static_cast<std::size_t>(tiles_m), [&](std::size_t ti) {
      const std::int64_t i0 = static_cast<std::int64_t>(ti) * kTile;
      const std::int64_t i1 = std::min(i0 + kTile, m);
      for (std::int64_t j0 = 0; j0 < n; j0 += kTile) {
        const std::int64_t j1 = std::min(j0 + kTile, n);
        for (std::int64_t k0 = 0; k0 < k; k0 += kTile) {
          const std::int64_t k1 = std::min(k0 + kTile, k);
          for (std::int64_t i = i0; i < i1; ++i) {
            for (std::int64_t j = j0; j < j1; ++j) {
              float sum = 0.0f;
              for (std::int64_t kk = k0; kk < k1; ++kk) {
                sum += load_a(i, kk) * load_b(kk, j);
              }
              c[i * ldc + j] += alpha * sum;
            }
          }
        }
      }
    });
    return Error::success();
  }
};

RefGemm& ref_gemm() {
  static RefGemm g;
  return g;
}

}  // namespace

IGemm* try_make_onednn();  // defined in gemm_onednn.cpp
IGemm* try_make_openblas();  // defined in gemm_openblas.cpp

bool gemm_provider_linked(GemmProvider p) noexcept {
  switch (p) {
    case GemmProvider::Ref:
      return true;
    case GemmProvider::OneDnn:
      return try_make_onednn() != nullptr;
    case GemmProvider::OpenBlas:
      return try_make_openblas() != nullptr;
    default:
      return false;
  }
}

IGemm* try_get_gemm(GemmProvider p) {
  switch (p) {
    case GemmProvider::Ref:
      return &ref_gemm();
    case GemmProvider::OneDnn:
      return try_make_onednn();
    case GemmProvider::OpenBlas:
      return try_make_openblas();
    default:
      return nullptr;
  }
}

std::vector<GemmProvider> linked_gemm_providers() {
  std::vector<GemmProvider> out;
  out.push_back(GemmProvider::Ref);
  if (gemm_provider_linked(GemmProvider::OneDnn)) out.push_back(GemmProvider::OneDnn);
  if (gemm_provider_linked(GemmProvider::OpenBlas)) out.push_back(GemmProvider::OpenBlas);
  return out;
}

const char* to_string(GemmProvider p) noexcept {
  switch (p) {
    case GemmProvider::OneDnn: return "onednn";
    case GemmProvider::OpenBlas: return "openblas";
    case GemmProvider::CudaCublasLt: return "cublaslt";
    default: return "ref";
  }
}

GemmRegistry& GemmRegistry::instance() {
  static GemmRegistry reg;
  return reg;
}

GemmRegistry::GemmRegistry() {
  if (const char* env = std::getenv("UAII_GEMM"); env && env[0]) {
    const std::string v(env);
    if (v == "onednn") preferred_ = GemmProvider::OneDnn;
    else if (v == "openblas") preferred_ = GemmProvider::OpenBlas;
    else preferred_ = GemmProvider::Ref;
  } else {
#if defined(UAII_HAVE_ONEDNN)
    preferred_ = GemmProvider::OneDnn;
#elif defined(UAII_HAVE_OPENBLAS)
    preferred_ = GemmProvider::OpenBlas;
#else
    preferred_ = GemmProvider::Ref;
#endif
  }
}

void GemmRegistry::set_preferred(GemmProvider p) { preferred_ = p; }

IGemm& GemmRegistry::active() {
  if (preferred_ == GemmProvider::OneDnn) {
    if (IGemm* g = try_make_onednn()) return *g;
  }
  if (preferred_ == GemmProvider::OpenBlas) {
    if (IGemm* g = try_make_openblas()) return *g;
  }
  if (preferred_ == GemmProvider::OneDnn || preferred_ == GemmProvider::OpenBlas) {
    // Soft fallback
  }
  return ref_gemm();
}

const IGemm& GemmRegistry::active() const {
  return const_cast<GemmRegistry*>(this)->active();
}

GemmProvider GemmRegistry::active_provider() const noexcept {
  return active().provider();
}

std::string GemmRegistry::describe() const {
  const IGemm& g = active();
  return std::string(g.name()) + " (" + to_string(g.provider()) + ")";
}

IGemm& default_gemm() { return GemmRegistry::instance().active(); }

}  // namespace kernels
}  // namespace uaii
