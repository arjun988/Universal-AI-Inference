#include "uaii/planner/memory_plan.hpp"

#include "uaii/ir/dtype.hpp"

#include <algorithm>
#include <sstream>

namespace uaii {
namespace planner {

Error build_memory_reuse_plan(const ir::Graph& graph,
                              const ir::ExecutionPlan& plan,
                              MemoryReusePlan* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "memory plan out null");
  }
  *out = MemoryReusePlan{};

  auto is_input = [&](TensorId id) {
    return std::find(graph.inputs.begin(), graph.inputs.end(), id) != graph.inputs.end();
  };
  auto is_output = [&](TensorId id) {
    return std::find(graph.outputs.begin(), graph.outputs.end(), id) != graph.outputs.end();
  };

  std::unordered_map<TensorId, TensorLifetime> lives;
  for (const auto& t : graph.tensors) {
    TensorLifetime life;
    life.tensor_id = t.id;
    life.bytes = ir::estimate_tensor_bytes(t);
    life.is_weight = t.is_weight;
    life.is_graph_io = is_input(t.id) || is_output(t.id);
    if (life.is_graph_io || life.is_weight) {
      life.first_use = 0;
      life.last_use = static_cast<int>(plan.ops.size());
    }
    lives[t.id] = life;
    out->naive_bytes += life.bytes;
    if (life.is_weight) out->weight_bytes += life.bytes;
  }

  for (int oi = 0; oi < static_cast<int>(plan.ops.size()); ++oi) {
    const auto& op = plan.ops[static_cast<std::size_t>(oi)];
    for (TensorId id : op.inputs) {
      auto it = lives.find(id);
      if (it == lives.end()) continue;
      if (it->second.first_use < 0 || oi < it->second.first_use) it->second.first_use = oi;
      if (oi > it->second.last_use) it->second.last_use = oi;
    }
    for (TensorId id : op.outputs) {
      auto it = lives.find(id);
      if (it == lives.end()) continue;
      if (it->second.first_use < 0) it->second.first_use = oi;
      if (oi > it->second.last_use) it->second.last_use = oi;
    }
  }

  // Assign dedicated slots for weights + graph IO
  int next_slot = 0;
  for (auto& kv : lives) {
    TensorLifetime& life = kv.second;
    out->lifetimes.push_back(life);
    if (life.is_weight || life.is_graph_io) {
      BufferSlot slot;
      slot.slot_id = next_slot++;
      slot.bytes = life.bytes;
      slot.tenants.push_back(life.tensor_id);
      out->tensor_to_slot[life.tensor_id] = slot.slot_id;
      out->slots.push_back(slot);
    }
  }

  // Greedy reuse for activation tensors
  struct FreeSlot {
    int slot_id = 0;
    std::uint64_t bytes = 0;
    int free_after = -1;
  };
  std::vector<FreeSlot> free_list;

  std::vector<TensorLifetime*> acts;
  for (auto& life : out->lifetimes) {
    if (!life.is_weight && !life.is_graph_io) acts.push_back(&life);
  }
  std::sort(acts.begin(), acts.end(), [](const TensorLifetime* a, const TensorLifetime* b) {
    if (a->first_use != b->first_use) return a->first_use < b->first_use;
    return a->tensor_id < b->tensor_id;
  });

  for (TensorLifetime* life : acts) {
    int chosen = -1;
    for (std::size_t i = 0; i < free_list.size(); ++i) {
      if (free_list[i].free_after < life->first_use &&
          free_list[i].bytes >= life->bytes) {
        chosen = static_cast<int>(i);
        break;
      }
    }
    if (chosen >= 0) {
      const int sid = free_list[static_cast<std::size_t>(chosen)].slot_id;
      out->tensor_to_slot[life->tensor_id] = sid;
      out->slots[static_cast<std::size_t>(sid)].tenants.push_back(life->tensor_id);
      if (life->bytes > out->slots[static_cast<std::size_t>(sid)].bytes) {
        out->slots[static_cast<std::size_t>(sid)].bytes = life->bytes;
      }
      free_list.erase(free_list.begin() + chosen);
      free_list.push_back(FreeSlot{sid, out->slots[static_cast<std::size_t>(sid)].bytes,
                                   life->last_use});
    } else {
      BufferSlot slot;
      slot.slot_id = next_slot++;
      slot.bytes = life->bytes;
      slot.tenants.push_back(life->tensor_id);
      out->tensor_to_slot[life->tensor_id] = slot.slot_id;
      out->slots.push_back(slot);
      free_list.push_back(FreeSlot{slot.slot_id, slot.bytes, life->last_use});
    }
  }

  // Peak = sum of slot bytes for non-weight slots + weights (resident)
  out->peak_bytes = 0;
  for (const auto& slot : out->slots) {
    out->peak_bytes += slot.bytes;
  }

  std::ostringstream oss;
  oss << "naive=" << out->naive_bytes << "B peak_reuse=" << out->peak_bytes
      << "B weights=" << out->weight_bytes << "B slots=" << out->slots.size()
      << "/" << lives.size();
  out->summary = oss.str();
  return Error::ok();
}

}  // namespace planner
}  // namespace uaii
