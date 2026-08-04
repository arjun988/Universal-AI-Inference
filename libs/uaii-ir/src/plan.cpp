#include "uaii/ir/plan.hpp"

#include "uaii/ir/dtype.hpp"

#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace uaii {
namespace ir {

Error build_execution_plan(const Graph& graph, ExecutionPlan* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "execution plan out is null");
  }
  *out = ExecutionPlan{};
  out->ir_version = graph.version;
  out->graph_name = graph.name;
  out->graph_inputs = graph.inputs;
  out->graph_outputs = graph.outputs;

  std::unordered_map<TensorId, const Tensor*> tensors;
  for (const auto& t : graph.tensors) {
    tensors[t.id] = &t;
  }

  std::unordered_set<TensorId> input_set(graph.inputs.begin(), graph.inputs.end());
  std::unordered_set<TensorId> output_set(graph.outputs.begin(), graph.outputs.end());

  for (const auto& t : graph.tensors) {
    MemoryPlanHint hint;
    hint.tensor_id = t.id;
    hint.name = t.name;
    hint.estimated_bytes = estimate_tensor_bytes(t);
    hint.is_graph_input = input_set.count(t.id) != 0;
    hint.is_graph_output = output_set.count(t.id) != 0;
    hint.is_weight = t.is_weight;
    out->memory_hints.push_back(std::move(hint));
  }

  if (graph.nodes.empty()) {
    return Error::ok();
  }

  std::unordered_map<TensorId, NodeId> producer;
  for (const auto& n : graph.nodes) {
    for (TensorId tid : n.outputs) {
      producer[tid] = n.id;
    }
  }

  std::unordered_map<NodeId, const Node*> nodes;
  std::unordered_map<NodeId, std::vector<NodeId>> adj;
  std::unordered_map<NodeId, int> indegree;
  std::unordered_map<NodeId, std::vector<NodeId>> deps;

  for (const auto& n : graph.nodes) {
    nodes[n.id] = &n;
    indegree[n.id] = 0;
  }

  for (const auto& n : graph.nodes) {
    std::unordered_set<NodeId> unique_deps;
    for (TensorId tin : n.inputs) {
      auto it = producer.find(tin);
      if (it == producer.end() || it->second == n.id) {
        continue;
      }
      unique_deps.insert(it->second);
    }
    for (NodeId pred : unique_deps) {
      adj[pred].push_back(n.id);
      indegree[n.id] += 1;
      deps[n.id].push_back(pred);
    }
  }

  std::queue<NodeId> q;
  for (const auto& kv : indegree) {
    if (kv.second == 0) {
      q.push(kv.first);
    }
  }

  std::vector<NodeId> order;
  while (!q.empty()) {
    const NodeId u = q.front();
    q.pop();
    order.push_back(u);
    for (NodeId v : adj[u]) {
      if (--indegree[v] == 0) {
        q.push(v);
      }
    }
  }

  if (order.size() != graph.nodes.size()) {
    return Error::make(ErrorCode::InvalidArgument,
                       "cannot build execution plan: graph has a cycle");
  }

  out->ops.reserve(order.size());
  for (NodeId id : order) {
    const Node* n = nodes[id];
    PlannedOp op;
    op.node_id = n->id;
    op.node_name = n->name;
    op.op_name = n->op_name;
    op.op_version = n->op_version;
    op.dependencies = deps[id];
    op.preferred_device = DeviceType::Cpu;
    op.inputs = n->inputs;
    op.outputs = n->outputs;
    out->ops.push_back(std::move(op));
  }

  return Error::ok();
}

std::string plan_to_text(const ExecutionPlan& plan) {
  std::ostringstream oss;
  oss << "ExecutionPlan graph='" << plan.graph_name << "' ir="
      << to_string(plan.ir_version) << " ops=" << plan.ops.size() << "\n";
  for (std::size_t i = 0; i < plan.ops.size(); ++i) {
    const auto& op = plan.ops[i];
    oss << "  [" << i << "] node=" << op.node_id << " " << op.op_name << "@"
        << op.op_version;
    if (!op.node_name.empty()) {
      oss << " name=" << op.node_name;
    }
    if (!op.dependencies.empty()) {
      oss << " deps=[";
      for (std::size_t d = 0; d < op.dependencies.size(); ++d) {
        if (d) oss << ",";
        oss << op.dependencies[d];
      }
      oss << "]";
    }
    oss << "\n";
  }
  return oss.str();
}

}  // namespace ir
}  // namespace uaii
