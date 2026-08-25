#include "uaii/planner/storage_plan.hpp"

#include "uaii/ir/dtype.hpp"
#include "uaii/quant/formats.hpp"

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

bool is_embed_name(const std::string& name) {
  const std::string n = lower_ascii(name);
  return n.find("embed") != std::string::npos || n.find("token_embd") != std::string::npos ||
         n.find("tok_embeddings") != std::string::npos || n.find("wte") != std::string::npos;
}

bool is_lm_head_name(const std::string& name) {
  const std::string n = lower_ascii(name);
  return n.find("lm_head") != std::string::npos || n == "output" ||
         n.find("output.weight") != std::string::npos;
}

bool is_norm_name(const std::string& name) {
  const std::string n = lower_ascii(name);
  return n.find("output_norm") != std::string::npos || n.find("rms_out") != std::string::npos ||
         n.find("ln_f") != std::string::npos || n.find("final_layernorm") != std::string::npos ||
         n.find("final_layer_norm") != std::string::npos || n == "norm" ||
         n == "model.norm.weight" || n == "model.norm.bias";
}

bool is_trunk_always_pin(const std::string& name) {
  return is_embed_name(name) || is_lm_head_name(name) || is_norm_name(name);
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

std::uint64_t weight_resident_bytes(const ir::Tensor& t) {
  const std::size_t n = ir::shape_numel(t.shape);
  if (t.is_weight && quant::is_gguf_block_quant(t.quant_format)) {
    return static_cast<std::uint64_t>(quant::packed_nbytes(t.quant_format, n));
  }
  return ir::estimate_tensor_bytes(t);
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
    p.bytes = weight_resident_bytes(t);
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

  const bool pressure = options.enable_streaming && options.ram_budget_bytes > 0 &&
                        out->total_weight_bytes > options.ram_budget_bytes;
  const bool expert_page = options.enable_streaming && expert_bytes > 0 &&
                           (pressure || options.expert_cache_bytes > 0);

  std::int64_t pin_layers = options.pin_layers;
  bool pin_embed = true;
  bool pin_lm = true;
  // Real laptop/desktop budgets: never pin more than the budget (kimi floor dial).
  const bool tight_budget = options.ram_budget_bytes >= (1ull << 20);

  auto is_layer_pinned = [&](const StoragePlacement& p) -> bool {
    if (pin_layers <= 0) return false;
    for (std::int64_t i = 0; i < pin_layers; ++i) {
      if (name_has_layer(p.name, i)) return true;
    }
    return false;
  };

  auto would_pin = [&](const StoragePlacement& p) -> bool {
    if (!p.expert && p.bytes > 0 &&
        (p.tier == StorageTier::Mmap || p.tier == StorageTier::Disk)) {
      if (is_norm_name(p.name)) return true;
      if (pin_embed && is_embed_name(p.name)) return true;
      if (pin_lm && is_lm_head_name(p.name)) return true;
      if (is_layer_pinned(p)) return true;
      if (!tight_budget && is_trunk_always_pin(p.name)) return true;
    }
    return false;
  };

  auto pin_and_staging = [&](std::uint64_t* pin_out, std::uint64_t* stage_out,
                             std::string* biggest_stream) {
    std::uint64_t pin = 0;
    std::uint64_t stage = 0;
    std::string big;
    for (const auto& p : out->placements) {
      if (p.bytes == 0) continue;
      if (p.tier != StorageTier::Mmap && p.tier != StorageTier::Disk) continue;
      if (p.expert) {
        if (p.bytes > stage) {
          stage = p.bytes;
          big = p.name;
        }
        continue;
      }
      if (would_pin(p)) {
        pin += p.bytes;
      } else if (p.bytes > stage) {
        stage = p.bytes;
        big = p.name;
      }
    }
    if (pin_out) *pin_out = pin;
    if (stage_out) *stage_out = stage;
    if (biggest_stream) *biggest_stream = std::move(big);
  };

  if (pressure || expert_page) {
    out->streaming_required = true;

    if (tight_budget) {
      // kimi floor dial: pin trunk first; expert LRU gets only leftover bytes.
      std::uint64_t pin = 0;
      std::uint64_t stage = 0;
      std::string biggest;
      for (int step = 0; step < 64; ++step) {
        pin_and_staging(&pin, &stage, &biggest);
        const std::uint64_t need = pin + stage * 2ull;
        if (need <= options.ram_budget_bytes) break;
        if (pin_layers > 0) {
          --pin_layers;
          continue;
        }
        if (pin_lm) {
          pin_lm = false;
          continue;
        }
        if (pin_embed) {
          pin_embed = false;
          continue;
        }
        break;
      }
      pin_and_staging(&pin, &stage, &biggest);
      const std::uint64_t need = pin + stage * 2ull;
      if (stage > 0 && need > options.ram_budget_bytes) {
        return Error::make(
            ErrorCode::InvalidArgument,
            "weight " + biggest + " is " + std::to_string(stage) +
                " bytes which exceeds the memory preset. Convert to GGUF Q4_K "
                "or use --preset desktop|server.");
      }
      const std::uint64_t leftover =
          options.ram_budget_bytes > need ? options.ram_budget_bytes - need : 0;
      if (options.expert_cache_bytes > leftover) {
        out->expert_cache_bytes = leftover;
      }
    }

    for (auto& p : out->placements) {
      if (p.bytes == 0) continue;
      if (p.tier != StorageTier::Mmap && p.tier != StorageTier::Disk) continue;
      const bool pin = would_pin(p) && !p.expert;
      if (p.expert || !pin) {
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

  std::uint64_t max_streamed = 0;
  for (const auto& p : out->placements) {
    if (p.stream) max_streamed = std::max(max_streamed, p.bytes);
  }
  out->staging_bytes = max_streamed > 0 ? max_streamed : max_weight;

  std::ostringstream oss;
  oss << "weights=" << out->total_weight_bytes << "B dense=" << dense_bytes
      << "B experts=" << expert_bytes << "B budget=" << out->ram_budget_bytes
      << "B staging=" << out->staging_bytes << "B expert_cache=" << out->expert_cache_bytes
      << "B streaming=" << (out->streaming_required ? "yes" : "no")
      << " pin_layers=" << pin_layers << " pin_embed=" << (pin_embed ? "yes" : "no")
      << " pin_lm_head=" << (pin_lm ? "yes" : "no") << " windows=" << out->windows.size()
      << " expert_windows=" << out->expert_windows;
  out->summary = oss.str();
  return Error::success();
}

}  // namespace planner
}  // namespace uaii
