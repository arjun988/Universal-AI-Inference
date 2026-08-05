#include "uaii/planner/optimize.hpp"

#include "uaii/planner/cache.hpp"

#include <sstream>

namespace uaii {
namespace planner {

Error optimize_graph(ir::Graph graph,
                     const OptimizeOptions& options,
                     OptimizeResult* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "optimize out null");
  }
  *out = OptimizeResult{};

  if (options.enable_fusion) {
    Error err = apply_fusion_passes(&graph, &out->fusion);
    if (!err.ok()) return err;
  } else {
    out->fusion.nodes_before = static_cast<int>(graph.nodes.size());
    out->fusion.nodes_after = out->fusion.nodes_before;
  }

  const std::string key = PlanCache::fingerprint(graph, options.enable_fusion);
  if (options.enable_plan_cache && PlanCache::instance().get(key, &out->plan)) {
    out->cache_hit = true;
  } else {
    Error err = ir::build_execution_plan(graph, &out->plan);
    if (!err.ok()) return err;
    if (options.enable_plan_cache) {
      PlanCache::instance().put(key, out->plan);
    }
  }

  if (options.enable_memory_reuse) {
    Error err = build_memory_reuse_plan(graph, out->plan, &out->memory);
    if (!err.ok()) return err;
  }

  if (options.enable_storage_plan) {
    StoragePlanOptions sopts;
    sopts.ram_budget_bytes = options.ram_budget_bytes;
    sopts.prefer_mmap = options.prefer_mmap;
    sopts.enable_streaming = options.enable_streaming;
    Error err = build_storage_plan(graph, sopts, &out->storage);
    if (!err.ok()) return err;
  }

  out->graph = std::move(graph);

  std::ostringstream oss;
  oss << "fusion nodes " << out->fusion.nodes_before << "→" << out->fusion.nodes_after
      << " (matmul_relu=" << out->fusion.matmul_relu_fused
      << ", identity=" << out->fusion.identity_removed << ")"
      << "; " << out->memory.summary << "; " << out->storage.summary
      << "; cache=" << (out->cache_hit ? "hit" : "miss");
  out->summary = oss.str();
  return Error::success();
}

}  // namespace planner
}  // namespace uaii
