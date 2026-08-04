#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/plan.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace planner {

struct TensorLifetime {
  TensorId tensor_id = 0;
  int first_use = -1;  // op index in plan
  int last_use = -1;
  std::uint64_t bytes = 0;
  bool is_weight = false;
  bool is_graph_io = false;
};

struct BufferSlot {
  int slot_id = 0;
  std::uint64_t bytes = 0;
  std::vector<TensorId> tenants;  // tensors that share this slot over time
};

struct MemoryReusePlan {
  std::vector<TensorLifetime> lifetimes;
  std::vector<BufferSlot> slots;
  std::unordered_map<TensorId, int> tensor_to_slot;
  std::uint64_t naive_bytes = 0;   // sum of all tensor sizes
  std::uint64_t peak_bytes = 0;    // with reuse (excl. weights if streamed)
  std::uint64_t weight_bytes = 0;
  std::string summary;
};

/// Lifetime analysis + greedy buffer reuse for non-weight activations.
[[nodiscard]] UAII_API Error build_memory_reuse_plan(const ir::Graph& graph,
                                                     const ir::ExecutionPlan& plan,
                                                     MemoryReusePlan* out);

}  // namespace planner
}  // namespace uaii
