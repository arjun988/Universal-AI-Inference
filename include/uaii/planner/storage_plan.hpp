#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/storage.hpp"
#include "uaii/ir/graph.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace uaii {
namespace planner {

struct StoragePlacement {
  TensorId tensor_id = 0;
  std::string name;
  StorageTier tier = StorageTier::Ram;
  bool stream = false;
  /// Routed MoE expert weight — never pin; LRU-capped at expert_cache_bytes.
  bool expert = false;
  std::uint64_t bytes = 0;
  std::string uri;  // weight_ref (file#tensor) when streamed
};

struct StreamingWindow {
  TensorId tensor_id = 0;
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
};

struct StoragePlan {
  std::vector<StoragePlacement> placements;
  std::vector<StreamingWindow> windows;
  std::uint64_t ram_budget_bytes = 0;
  std::uint64_t total_weight_bytes = 0;
  std::uint64_t staging_bytes = 0;  // max single streamed weight
  std::uint64_t expert_cache_bytes = 0;
  std::size_t expert_windows = 0;
  bool streaming_required = false;
  std::string summary;
};

struct StoragePlanOptions {
  std::uint64_t ram_budget_bytes = 0;  // 0 = no streaming pressure
  bool prefer_mmap = true;
  bool enable_streaming = true;
  /// Preferred pinned trunk depth. Under a real RAM budget (>= 1 MiB) the planner
  /// drops layers, then lm_head, then embed until pin + 2×staging fits.
  std::int64_t pin_layers = 0;
  /// Expert LRU budget (bytes). 0 = experts still stream but are not cached across tokens.
  /// Tight budgets assign leftover RAM after the trunk, never at the trunk's expense.
  std::uint64_t expert_cache_bytes = 0;
};

/// True for Mixtral/Qwen-MoE expert banks and per-expert HF tensors.
[[nodiscard]] UAII_API bool is_moe_expert_weight_name(const std::string& name) noexcept;

[[nodiscard]] UAII_API Error build_storage_plan(const ir::Graph& graph,
                                                const StoragePlanOptions& options,
                                                StoragePlan* out);

}  // namespace planner
}  // namespace uaii
