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
  std::uint64_t bytes = 0;
  std::string uri;  // weight_ref path when streamed
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
  bool streaming_required = false;
  std::string summary;
};

struct StoragePlanOptions {
  std::uint64_t ram_budget_bytes = 0;  // 0 = no streaming pressure
  bool prefer_mmap = true;
  bool enable_streaming = true;
};

[[nodiscard]] UAII_API Error build_storage_plan(const ir::Graph& graph,
                                                const StoragePlanOptions& options,
                                                StoragePlan* out);

}  // namespace planner
}  // namespace uaii
