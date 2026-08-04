#include "uaii/planner/fusion.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace uaii {
namespace planner {
namespace {

int count_consumers(const ir::Graph& g, TensorId tid) {
  int n = 0;
  for (const auto& node : g.nodes) {
    for (TensorId in : node.inputs) {
      if (in == tid) ++n;
    }
  }
  for (TensorId out : g.outputs) {
    if (out == tid) ++n;
  }
  return n;
}

bool is_graph_io(const ir::Graph& g, TensorId tid) {
  for (TensorId id : g.inputs) {
    if (id == tid) return true;
  }
  for (TensorId id : g.outputs) {
    if (id == tid) return true;
  }
  return false;
}

}  // namespace

Error apply_fusion_passes(ir::Graph* graph, FusionStats* stats) {
  if (graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "fusion graph null");
  }
  FusionStats local;
  FusionStats* s = stats ? stats : &local;
  *s = FusionStats{};
  s->nodes_before = static_cast<int>(graph->nodes.size());

  // Pass 1: remove Identity (rewire consumers to Identity input)
  {
    std::vector<ir::Node> kept;
    kept.reserve(graph->nodes.size());
    std::unordered_map<TensorId, TensorId> replace;
    for (const auto& node : graph->nodes) {
      if (node.op_name == "Identity" && node.inputs.size() == 1 &&
          node.outputs.size() == 1 && !is_graph_io(*graph, node.outputs[0])) {
        replace[node.outputs[0]] = node.inputs[0];
        ++s->identity_removed;
        continue;
      }
      kept.push_back(node);
    }
    if (!replace.empty()) {
      for (auto& node : kept) {
        for (auto& in : node.inputs) {
          auto it = replace.find(in);
          if (it != replace.end()) in = it->second;
        }
      }
      for (auto& out : graph->outputs) {
        auto it = replace.find(out);
        if (it != replace.end()) out = it->second;
      }
      // Drop unused intermediate tensors produced by removed Identity
      std::unordered_set<TensorId> live;
      for (TensorId id : graph->inputs) live.insert(id);
      for (TensorId id : graph->outputs) live.insert(id);
      for (const auto& n : kept) {
        for (TensorId id : n.inputs) live.insert(id);
        for (TensorId id : n.outputs) live.insert(id);
      }
      std::vector<ir::Tensor> tensors;
      for (const auto& t : graph->tensors) {
        if (live.count(t.id) != 0) tensors.push_back(t);
      }
      graph->tensors = std::move(tensors);
      graph->nodes = std::move(kept);
    }
  }

  // Pass 2: MatMul → Relu fusion when MatMul output has single Relu consumer
  {
    std::unordered_map<TensorId, const ir::Node*> producer;
    for (const auto& n : graph->nodes) {
      for (TensorId o : n.outputs) producer[o] = &n;
    }

    std::unordered_set<NodeId> remove;
    std::vector<ir::Node> fused_nodes;

    for (const auto& relu : graph->nodes) {
      if (relu.op_name != "Relu" || relu.inputs.size() != 1 || relu.outputs.size() != 1) {
        continue;
      }
      auto pit = producer.find(relu.inputs[0]);
      if (pit == producer.end()) continue;
      const ir::Node* mm = pit->second;
      if (mm->op_name != "MatMul" || mm->outputs.size() != 1) continue;
      if (count_consumers(*graph, mm->outputs[0]) != 1) continue;
      if (is_graph_io(*graph, mm->outputs[0])) continue;

      ir::Node fused = *mm;
      fused.name = mm->name + "_relu";
      fused.op_name = "MatMulRelu";
      fused.outputs = relu.outputs;
      fused_nodes.push_back(fused);
      remove.insert(mm->id);
      remove.insert(relu.id);
      ++s->matmul_relu_fused;
    }

    if (!remove.empty()) {
      std::vector<ir::Node> kept;
      for (const auto& n : graph->nodes) {
        if (remove.count(n.id) != 0) continue;
        kept.push_back(n);
      }
      for (auto& f : fused_nodes) kept.push_back(std::move(f));
      // Drop dead intermediate tensors
      std::unordered_set<TensorId> live;
      for (TensorId id : graph->inputs) live.insert(id);
      for (TensorId id : graph->outputs) live.insert(id);
      for (const auto& n : kept) {
        for (TensorId id : n.inputs) live.insert(id);
        for (TensorId id : n.outputs) live.insert(id);
      }
      std::vector<ir::Tensor> tensors;
      for (const auto& t : graph->tensors) {
        if (live.count(t.id) != 0) tensors.push_back(t);
      }
      graph->tensors = std::move(tensors);
      graph->nodes = std::move(kept);
    }
  }

  s->nodes_after = static_cast<int>(graph->nodes.size());
  return Error::ok();
}

}  // namespace planner
}  // namespace uaii
