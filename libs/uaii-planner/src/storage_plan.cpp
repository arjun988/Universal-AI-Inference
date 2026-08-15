#include "uaii/planner/storage_plan.hpp"

#include "uaii/ir/dtype.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace uaii {
namespace planner {
namespace {

std::string lower_ascii(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool is_trunk_always_pin(const std::string& name) {
  const std::string n = lower_ascii(name);
  if (n.find("embed") != std::string::npos || n.find("token_embd") != std::string::npos ||
      n.find("tok_embeddings") != std::string::npos || n.find("wte") != std::string::npos) {
    return true;
  }
  if (n.find("lm_head") != std::string::npos || n == "output" || n.find("output.weight") != std::string::npos) {
    return true;
  }
  if (n.find("output_norm") != std::string::npos || n.find("rms_out") != std::string::npos ||
      n.find("ln_f") != std::string::npos || n.find("final_layernorm") != std::string::npos ||
      n.find("final_layer_norm") != std::string::npos || n == "norm" ||
      n == "model.norm.weight" || n == "model.norm.bias") {
    return true;
  }
  return false;
}

bool name_has_layer(const std::string& name, std::int64_t i) {
  const std::string si = std::to_string(i);
  const std::string pats[] = {
      "blk." + si + ".",
      "l" + si + ".",
      "model.layers." + si + ".",
      "model.decoder.layers." + si + ".",
      "transformer.h." + si + ".",
      "transformer.layers." + si + ".",
      "gpt_neox.layers." + si + ".",
  };
  for (const auto& p : pats) {
    if (name.find(p) != std::string::npos) return true;
  }
  return false;
}

}  // namespace

bool is_moe_expert_weight_name(const std::string& name) noexcept {
  if (name.find("ffn_gate_exps") != std::string::npos ||
      name.find("ffn_up_exps") != std::string::npos ||
      name.find("ffn_down_exps") != std::string::npos) {
    return true;
  }
  if (name.find("block_sparse_moe.experts") != std::string::npos) return true;
  if (name.find(".exp.") != std::string::npos) return true;
  if (name.find(".experts.") != std::string::npos) return true;
  return false;
}

Error build_storage_plan(const ir::Graph& graph,
                         const StoragePlanOptions& options,
                         StoragePlan* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "storage plan out null");
  }
  *out = StoragePlan{};
  out->ram_budget_bytes = options.ram_budget_bytes;
  out->expert_cache_bytes = options.expert_cache_bytes;

  std::uint64_t max_weight = 0;
  std::uint64_t dense_bytes = 0;
  std::uint64_t expert_bytes = 0;
  for (const auto& t : graph.tensors) {
    StoragePlacement p;
    p.tensor_id = t.id;
    p.name = t.name;
    p.bytes = ir::estimate_tensor_bytes(t);
    p.expert = t.is_weight && is_moe_expert_weight_name(t.name);
    if (t.is_weight) {
      out->total_weight_bytes += p.bytes;
      max_weight = std::max(max_weight, p.bytes);
      p.uri = t.weight_ref;
      p.tier = options.prefer_mmap ? StorageTier::Mmap : StorageTier::Disk;
      p.stream = false;
      if (p.expert) {
        expert_bytes += p.bytes;
      } else {
        dense_bytes += p.bytes;
      }
    } else {
      p.tier = StorageTier::Ram;
    }
    out->placements.push_back(p);
  }

  out->staging_bytes = max_weight;

  const bool pressure = options.enable_streaming && options.ram_budget_bytes > 0 &&
                        out->total_weight_bytes > options.ram_budget_bytes;
  // Experts are never fully resident when an expert cache budget is set (kimi DNA).
  const bool expert_page = options.enable_streaming && expert_bytes > 0 &&
                           (pressure || options.expert_cache_bytes > 0);

  if (pressure || expert_page) {
    out->streaming_required = true;
    auto is_pinned_dense = [&](const StoragePlacement& p) -> bool {
      if (p.expert) return false;
      if (is_trunk_always_pin(p.name)) return true;
      if (options.pin_layers <= 0) return false;
      for (std::int64_t i = 0; i < options.pin_layers; ++i) {
        if (name_has_layer(p.name, i)) return true;
      }
      return false;
    };

    for (auto& p : out->placements) {
      if (p.bytes == 0) continue;
      if (p.tier != StorageTier::Mmap && p.tier != StorageTier::Disk) continue;
      if (p.expert || !is_pinned_dense(p)) {
        p.stream = true;
        StreamingWindow w;
        w.tensor_id = p.tensor_id;
        w.offset = 0;
        w.size = p.bytes;
        out->windows.push_back(w);
        if (p.expert) ++out->expert_windows;
      } else {
        p.stream = false;
        p.tier = StorageTier::Ram;
      }
    }
  }

  std::ostringstream oss;
  oss << "weights=" << out->total_weight_bytes << "B dense=" << dense_bytes
      << "B experts=" << expert_bytes << "B budget=" << out->ram_budget_bytes
      << "B staging=" << out->staging_bytes << "B expert_cache=" << out->expert_cache_bytes
      << "B streaming=" << (out->streaming_required ? "yes" : "no")
      << " pin_layers=" << options.pin_layers << " windows=" << out->windows.size()
      << " expert_windows=" << out->expert_windows;
  out->summary = oss.str();
  return Error::success();
}

}  // namespace planner
}  // namespace uaii
