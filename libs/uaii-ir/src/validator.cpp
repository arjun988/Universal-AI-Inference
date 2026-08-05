#include "uaii/ir/validator.hpp"

#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace uaii {
namespace ir {
namespace {

void add_issue(ValidationResult* result,
               ValidationSeverity severity,
               std::string code,
               std::string message,
               NodeId node_id = 0,
               TensorId tensor_id = 0) {
  ValidationIssue issue;
  issue.severity = severity;
  issue.code = std::move(code);
  issue.message = std::move(message);
  issue.node_id = node_id;
  issue.tensor_id = tensor_id;
  result->issues.push_back(std::move(issue));
}

bool arity_ok(int count, int min_v, int max_v) {
  if (min_v >= 0 && count < min_v) {
    return false;
  }
  if (max_v >= 0 && count > max_v) {
    return false;
  }
  return true;
}

}  // namespace

ValidationResult validate_graph(const Graph& graph,
                                const OperatorRegistry& registry,
                                const ValidationOptions& options) {
  ValidationResult result;

  if (auto err = check_compatible(graph.version); !err.ok()) {
    add_issue(&result, ValidationSeverity::Error, "ir.version", err.message());
  }

  if (graph.name.empty()) {
    add_issue(&result, ValidationSeverity::Warning, "graph.name",
              "graph name is empty");
  }

  std::unordered_set<TensorId> tensor_ids;
  std::unordered_map<std::string, TensorId> tensor_names;
  for (const auto& t : graph.tensors) {
    if (t.id == 0) {
      add_issue(&result, ValidationSeverity::Error, "tensor.id",
                "tensor id must be non-zero", 0, t.id);
      continue;
    }
    if (!tensor_ids.insert(t.id).second) {
      add_issue(&result, ValidationSeverity::Error, "tensor.duplicate_id",
                "duplicate tensor id " + std::to_string(t.id), 0, t.id);
    }
    if (!t.name.empty()) {
      auto [it, inserted] = tensor_names.emplace(t.name, t.id);
      if (!inserted) {
        add_issue(&result, ValidationSeverity::Error, "tensor.duplicate_name",
                  "duplicate tensor name '" + t.name + "'", 0, t.id);
      }
    }
    if (options.require_dtypes && t.dtype == DType::Unknown) {
      add_issue(&result, ValidationSeverity::Error, "tensor.dtype",
                "tensor '" + t.name + "' has unknown dtype", 0, t.id);
    }
    if (options.require_shapes && t.shape.dims.empty()) {
      add_issue(&result, ValidationSeverity::Error, "tensor.shape",
                "tensor '" + t.name + "' has empty shape", 0, t.id);
    }
  }

  auto require_tensor = [&](TensorId id, const char* where, NodeId node_id) {
    if (tensor_ids.find(id) == tensor_ids.end()) {
      add_issue(&result, ValidationSeverity::Error, "tensor.missing",
                std::string(where) + " references missing tensor id " +
                    std::to_string(id),
                node_id, id);
    }
  };

  for (TensorId id : graph.inputs) {
    require_tensor(id, "graph.inputs", 0);
  }
  for (TensorId id : graph.outputs) {
    require_tensor(id, "graph.outputs", 0);
  }
  if (graph.outputs.empty()) {
    add_issue(&result, ValidationSeverity::Error, "graph.outputs",
              "graph must declare at least one output");
  }

  std::unordered_set<NodeId> node_ids;
  std::unordered_map<TensorId, NodeId> tensor_producer;
  for (const auto& n : graph.nodes) {
    if (n.id == 0) {
      add_issue(&result, ValidationSeverity::Error, "node.id",
                "node id must be non-zero", n.id);
      continue;
    }
    if (!node_ids.insert(n.id).second) {
      add_issue(&result, ValidationSeverity::Error, "node.duplicate_id",
                "duplicate node id " + std::to_string(n.id), n.id);
    }
    if (n.op_name.empty()) {
      add_issue(&result, ValidationSeverity::Error, "node.op",
                "node missing op_name", n.id);
    }

    const OpSchema* schema = registry.find_schema(n.op_name, n.op_version);
    if (schema == nullptr) {
      if (options.allow_unknown_ops) {
        add_issue(&result, ValidationSeverity::Warning, "op.unknown",
                  "unknown operator " + n.op_name + "@" + n.op_version, n.id);
      } else {
        add_issue(&result, ValidationSeverity::Error, "op.unknown",
                  "unknown operator " + n.op_name + "@" + n.op_version, n.id);
      }
    } else {
      const int in_count = static_cast<int>(n.inputs.size());
      const int out_count = static_cast<int>(n.outputs.size());
      if (!arity_ok(in_count, schema->min_inputs, schema->max_inputs)) {
        add_issue(&result, ValidationSeverity::Error, "op.arity_inputs",
                  "operator " + n.op_name + " input arity mismatch", n.id);
      }
      if (!arity_ok(out_count, schema->min_outputs, schema->max_outputs)) {
        add_issue(&result, ValidationSeverity::Error, "op.arity_outputs",
                  "operator " + n.op_name + " output arity mismatch", n.id);
      }
    }

    for (TensorId id : n.inputs) {
      require_tensor(id, "node.inputs", n.id);
    }
    for (TensorId id : n.outputs) {
      require_tensor(id, "node.outputs", n.id);
      auto [it, inserted] = tensor_producer.emplace(id, n.id);
      if (!inserted) {
        add_issue(&result, ValidationSeverity::Error, "tensor.multi_producer",
                  "tensor id " + std::to_string(id) +
                      " produced by multiple nodes",
                  n.id, id);
      }
    }
  }

  // Graph inputs should not be produced by nodes.
  for (TensorId id : graph.inputs) {
    if (tensor_producer.find(id) != tensor_producer.end()) {
      add_issue(&result, ValidationSeverity::Error, "graph.input_produced",
                "graph input tensor " + std::to_string(id) +
                    " is also produced by a node",
                0, id);
    }
  }

  if (options.check_cycles && !graph.nodes.empty()) {
    std::unordered_map<NodeId, std::vector<NodeId>> adj;
    std::unordered_map<NodeId, int> indegree;
    for (const auto& n : graph.nodes) {
      indegree[n.id] = 0;
    }
    for (const auto& n : graph.nodes) {
      for (TensorId tin : n.inputs) {
        auto it = tensor_producer.find(tin);
        if (it == tensor_producer.end()) {
          continue;  // graph input / weight
        }
        const NodeId pred = it->second;
        if (pred == n.id) {
          add_issue(&result, ValidationSeverity::Error, "graph.self_cycle",
                    "node " + std::to_string(n.id) + " consumes its own output",
                    n.id);
          continue;
        }
        adj[pred].push_back(n.id);
        indegree[n.id] += 1;
      }
    }

    std::queue<NodeId> q;
    for (const auto& kv : indegree) {
      if (kv.second == 0) {
        q.push(kv.first);
      }
    }
    std::size_t seen = 0;
    while (!q.empty()) {
      const NodeId u = q.front();
      q.pop();
      ++seen;
      for (NodeId v : adj[u]) {
        if (--indegree[v] == 0) {
          q.push(v);
        }
      }
    }
    if (seen != graph.nodes.size()) {
      add_issue(&result, ValidationSeverity::Error, "graph.cycle",
                "graph contains a cycle");
    }
  }

  // Shape / dtype inference checks for core ops (fail closed on mismatches).
  if (options.require_shapes) {
    std::unordered_map<TensorId, const Tensor*> by_id;
    for (const auto& t : graph.tensors) by_id[t.id] = &t;
    for (const auto& n : graph.nodes) {
      auto T = [&](TensorId id) -> const Tensor* {
        auto it = by_id.find(id);
        return it == by_id.end() ? nullptr : it->second;
      };
      if (n.op_name == "MatMul" || n.op_name == "MatMulRelu") {
        if (n.inputs.size() == 2 && n.outputs.size() == 1) {
          const Tensor* a = T(n.inputs[0]);
          const Tensor* b = T(n.inputs[1]);
          const Tensor* c = T(n.outputs[0]);
          bool ta = false, tb = false;
          for (const auto& at : n.attributes) {
            if (at.key == "transpose_a" && at.type == AttributeType::Bool) {
              ta = std::get<bool>(at.value);
            }
            if (at.key == "transpose_b" && at.type == AttributeType::Bool) {
              tb = std::get<bool>(at.value);
            }
          }
          if (a && b && c && a->shape.dims.size() == 2 && b->shape.dims.size() == 2 &&
              c->shape.dims.size() == 2) {
            const std::int64_t a0 = ta ? a->shape.dims[1] : a->shape.dims[0];
            const std::int64_t a1 = ta ? a->shape.dims[0] : a->shape.dims[1];
            const std::int64_t b0 = tb ? b->shape.dims[1] : b->shape.dims[0];
            const std::int64_t b1 = tb ? b->shape.dims[0] : b->shape.dims[1];
            if (a1 != b0) {
              add_issue(&result, ValidationSeverity::Error, "shape.matmul",
                        "MatMul inner dims mismatch", n.id);
            }
            if (c->shape.dims[0] != a0 || c->shape.dims[1] != b1) {
              add_issue(&result, ValidationSeverity::Error, "shape.matmul_out",
                        "MatMul output shape mismatch", n.id);
            }
          }
        }
      }
      if (n.op_name == "Attention" && n.inputs.size() >= 3) {
        // num_heads must divide last dim of Q when present
        std::int64_t heads = 0;
        for (const auto& a : n.attributes) {
          if (a.key == "num_heads" && a.type == AttributeType::Int) {
            heads = std::get<std::int64_t>(a.value);
          }
        }
        const Tensor* q = T(n.inputs[0]);
        if (heads > 0 && q && !q->shape.dims.empty()) {
          const std::int64_t dim = q->shape.dims.back();
          if (dim % heads != 0) {
            add_issue(&result, ValidationSeverity::Error, "shape.attention_heads",
                      "Attention dim not divisible by num_heads", n.id);
          }
        }
      }
      if (n.op_name == "Add" || n.op_name == "Mul") {
        if (n.inputs.size() == 2 && n.outputs.size() == 1) {
          const Tensor* a = T(n.inputs[0]);
          const Tensor* b = T(n.inputs[1]);
          const Tensor* c = T(n.outputs[0]);
          if (a && b && c && a->shape.dims == b->shape.dims &&
              a->shape.dims != c->shape.dims) {
            add_issue(&result, ValidationSeverity::Error, "shape.elementwise",
                      n.op_name + " output shape mismatch", n.id);
          }
        }
      }
    }
  }

  return result;
}

Error validate_graph_error(const Graph& graph,
                           const OperatorRegistry& registry,
                           const ValidationOptions& options) {
  const ValidationResult result = validate_graph(graph, registry, options);
  if (result.ok()) {
    return Error::success();
  }
  std::ostringstream oss;
  oss << "graph validation failed with " << result.error_count() << " error(s):";
  for (const auto& issue : result.issues) {
    if (issue.severity != ValidationSeverity::Error) {
      continue;
    }
    oss << " [" << issue.code << "] " << issue.message << ";";
  }
  return Error::make(ErrorCode::InvalidArgument, oss.str());
}

}  // namespace ir
}  // namespace uaii
