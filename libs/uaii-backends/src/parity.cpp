#include "uaii/backends/parity.hpp"

#include <cmath>
#include <limits>

namespace uaii {
namespace backends {

Error compare_f32_buffers(const float* a,
                          const float* b,
                          std::size_t n,
                          const ParityPolicy& policy,
                          ParityTensorDiff* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "parity out null");
  }
  if (a == nullptr || b == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "parity buffers null");
  }

  out->max_abs_diff = 0;
  out->max_rel_diff = 0;
  out->ok = true;

  for (std::size_t i = 0; i < n; ++i) {
    const float x = a[i];
    const float y = b[i];
    if (policy.require_finite && (!std::isfinite(x) || !std::isfinite(y))) {
      out->ok = false;
      out->max_abs_diff = std::numeric_limits<float>::infinity();
      return Error::success();
    }
    const float abs_diff = std::fabs(x - y);
    const float denom = std::fmax(std::fabs(x), std::fabs(y));
    const float rel_diff = (denom > 0.f) ? (abs_diff / denom) : abs_diff;
    if (abs_diff > out->max_abs_diff) {
      out->max_abs_diff = abs_diff;
    }
    if (rel_diff > out->max_rel_diff) {
      out->max_rel_diff = rel_diff;
    }
    const float tol = policy.atol + policy.rtol * denom;
    if (abs_diff > tol) {
      out->ok = false;
    }
  }
  return Error::success();
}

}  // namespace backends
}  // namespace uaii
