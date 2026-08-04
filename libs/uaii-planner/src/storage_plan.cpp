#include "uaii/planner/storage_plan.hpp"

#include "uaii/ir/dtype.hpp"

#include <algorithm>
#include <sstream>

namespace uaii {
namespace planner {

Error build_storage_plan(const ir::Graph& graph,
                         const StoragePlanOptions& options,
                         StoragePlan* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "storage plan out null");
  }
  *out = StoragePlan{};
  out->ram_budget_bytes = options.ram_budget_bytes;

  std::uint64_t max_weight = 0;
  for (const auto& t : graph.tensors) {
    StoragePlacement p;
    p.tensor_id = t.id;
    p.name = t.name;
    p.bytes = ir::estimate_tensor_bytes(t);
    if (t.is_weight) {
      out->total_weight_bytes += p.bytes;
      max_weight = std::max(max_weight, p.bytes);
      p.uri = t.weight_ref;
      p.tier = options.prefer_mmap ? StorageTier::Mmap : StorageTier::Disk;
      p.stream = false;
    } else {
      p.tier = StorageTier::Ram;
    }
    out->placements.push_back(p);
  }

  out->staging_bytes = max_weight;
  if (options.enable_streaming && options.ram_budget_bytes > 0 &&
      out->total_weight_bytes > options.ram_budget_bytes) {
    out->streaming_required = true;
    // Stage one weight at a time; activations still need RAM separately.
    for (auto& p : out->placements) {
      if (p.bytes == 0) continue;
      if (p.tier == StorageTier::Mmap || p.tier == StorageTier::Disk) {
        p.stream = true;
        StreamingWindow w;
        w.tensor_id = p.tensor_id;
        w.offset = 0;
        w.size = p.bytes;
        out->windows.push_back(w);
      }
    }
  }

  std::ostringstream oss;
  oss << "weights=" << out->total_weight_bytes << "B budget=" << out->ram_budget_bytes
      << "B staging=" << out->staging_bytes
      << "B streaming=" << (out->streaming_required ? "yes" : "no")
      << " windows=" << out->windows.size();
  out->summary = oss.str();
  return Error::ok();
}

}  // namespace planner
}  // namespace uaii
