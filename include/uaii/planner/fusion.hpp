#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/graph.hpp"

#include <string>
#include <vector>

namespace uaii {
namespace planner {

struct FusionStats {
  int identity_removed = 0;
  int matmul_relu_fused = 0;
  int nodes_before = 0;
  int nodes_after = 0;
};

/// Rewrite graph in place: drop Identity; fuse MatMul→Relu into MatMulRelu.
[[nodiscard]] UAII_API Error apply_fusion_passes(ir::Graph* graph, FusionStats* stats);

}  // namespace planner
}  // namespace uaii
