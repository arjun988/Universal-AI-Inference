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
    auto is_pinned_layer_weight = [&](const std::string& name) -> bool {
      if (options.pin_layers <= 0) return false;
      // Always pin embeddings / final norm / lm_head when present.
      if (name.find("embed") != std::string::npos || name.find("token_embd") != std::string::npos ||
          name == "lm_head" || name.find("lm_head") != std::string::npos ||
          name.find("output_norm") != std::string::npos || name == "norm" ||
          name.find("rms_out") != std::string::npos) {
        return true;
      }
      for (std::int64_t i = 0; i < options.pin_layers; ++i) {
        const std::string blk = "blk." + std::to_string(i) + ".";
        const std::string lay = "l" + std::to_string(i) + ".";
        const std::string hf = "model.layers." + std::to_string(i) + ".";
        if (name.find(blk) != std::string::npos || name.find(lay) != std::string::npos ||
            name.find(hf) != std::string::npos) {
          return true;
        }
      }
      return false;
    };

    // Stage one weight at a time; pin prefix layers + emb/lm when pin_layers > 0.
    for (auto& p : out->placements) {
      if (p.bytes == 0) continue;
      if (p.tier == StorageTier::Mmap || p.tier == StorageTier::Disk) {
        if (is_pinned_layer_weight(p.name)) {
          p.stream = false;
          p.tier = StorageTier::Ram;
          continue;
        }
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
      << " pin_layers=" << options.pin_layers
      << " windows=" << out->windows.size();
  out->summary = oss.str();
  return Error::success();
}

}  // namespace planner
}  // namespace uaii
