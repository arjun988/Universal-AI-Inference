#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace uaii {
namespace backends {

struct ParityPolicy {
  float atol = 1e-5f;
  float rtol = 1e-4f;
  bool require_finite = true;
  /// Compare only graph outputs when true; otherwise all non-weight tensors.
  bool outputs_only = true;
};

struct ParityTensorDiff {
  std::string name;
  float max_abs_diff = 0;
  float max_rel_diff = 0;
  bool ok = true;
};

struct ParityReport {
  bool ok = false;
  std::string backend_a;
  std::string backend_b;
  std::string message;
  std::vector<ParityTensorDiff> diffs;
};

[[nodiscard]] UAII_API Error compare_f32_buffers(const float* a,
                                                 const float* b,
                                                 std::size_t n,
                                                 const ParityPolicy& policy,
                                                 ParityTensorDiff* out);

}  // namespace backends
}  // namespace uaii
