#pragma once

#include "uaii/export.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/plan.hpp"
#include "uaii/planner/fusion.hpp"
#include "uaii/planner/memory_plan.hpp"
#include "uaii/planner/storage_plan.hpp"

#include <cstdint>
#include <string>

namespace uaii {
namespace planner {

struct OptimizeOptions {
  bool enable_fusion = true;
  bool enable_memory_reuse = true;
  bool enable_storage_plan = true;
  bool enable_plan_cache = true;
  bool enable_streaming = false;
  std::uint64_t ram_budget_bytes = 0;
  bool prefer_mmap = true;
  std::int64_t pin_layers = 0;
  std::uint64_t expert_cache_bytes = 0;
  /// 1 = phone single staging buffer. 2 = double buffer. 0 = 2.
  int staging_slots = 2;
};

struct OptimizeResult {
  ir::Graph graph;  // possibly rewritten
  ir::ExecutionPlan plan;
  FusionStats fusion;
  MemoryReusePlan memory;
  StoragePlan storage;
  bool cache_hit = false;
  std::string summary;
};

/// Full Phase 6 optimization pipeline: fusion → plan → memory → storage.
[[nodiscard]] UAII_API Error optimize_graph(ir::Graph graph,
                                            const OptimizeOptions& options,
                                            OptimizeResult* out);

}  // namespace planner
}  // namespace uaii
