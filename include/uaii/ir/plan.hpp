#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/types.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/version.hpp"

#include <string>
#include <vector>

namespace uaii {
namespace ir {

/// Pre-scheduler execution plan entry (Phase 2 data structures).
struct PlannedOp {
  NodeId node_id = 0;
  std::string node_name;
  std::string op_name;
  std::string op_version;
  /// Topological predecessors within this plan.
  std::vector<NodeId> dependencies;
  /// Filled by planner/backends in later phases.
  DeviceType preferred_device = DeviceType::Cpu;
  std::string selected_kernel;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
};

struct MemoryPlanHint {
  TensorId tensor_id = 0;
  std::string name;
  /// Estimated bytes when shape/dtype known; 0 if dynamic/unknown.
  std::uint64_t estimated_bytes = 0;
  bool is_graph_input = false;
  bool is_graph_output = false;
  bool is_weight = false;
};

/// Execution plan derived from a validated IR graph (not device-finalized).
struct ExecutionPlan {
  IrVersion ir_version = kCurrentIrVersion;
  std::string graph_name;
  std::vector<PlannedOp> ops;  // topological order
  std::vector<MemoryPlanHint> memory_hints;
  std::vector<TensorId> graph_inputs;
  std::vector<TensorId> graph_outputs;
};

/// Build a topological execution plan from a graph.
/// Does not require validation, but callers should validate first.
[[nodiscard]] UAII_API Error build_execution_plan(const Graph& graph, ExecutionPlan* out);

[[nodiscard]] UAII_API std::string plan_to_text(const ExecutionPlan& plan);

}  // namespace ir
}  // namespace uaii
