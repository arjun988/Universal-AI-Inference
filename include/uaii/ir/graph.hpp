#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/node.hpp"
#include "uaii/ir/tensor.hpp"
#include "uaii/ir/version.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace ir {

/// UAII Intermediate Representation graph — the single source of truth.
struct Graph {
  IrVersion version = kCurrentIrVersion;
  std::string name;
  std::string producer;
  std::string domain = "uaii";
  std::vector<Tensor> tensors;
  std::vector<Node> nodes;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::unordered_map<std::string, std::string> metadata;

  [[nodiscard]] const Tensor* find_tensor(TensorId id) const;
  [[nodiscard]] Tensor* find_tensor(TensorId id);
  [[nodiscard]] const Node* find_node(NodeId id) const;

  [[nodiscard]] bool has_tensor(TensorId id) const { return find_tensor(id) != nullptr; }
  [[nodiscard]] bool has_node(NodeId id) const { return find_node(id) != nullptr; }

  void clear();
};

/// Convenience builders for hand-authored graphs in tests/examples/tools.
class UAII_API GraphBuilder {
 public:
  explicit GraphBuilder(std::string name);

  GraphBuilder& set_producer(std::string producer);
  GraphBuilder& set_domain(std::string domain);
  GraphBuilder& set_version(IrVersion version);
  GraphBuilder& set_metadata(std::string key, std::string value);

  TensorId add_tensor(std::string name, DType dtype, Shape shape,
                      StorageHint hint = StorageHint::Unspecified);
  TensorId add_weight(std::string name, DType dtype, Shape shape, std::string weight_ref);

  NodeId add_node(std::string name,
                  std::string op_name,
                  std::string op_version,
                  std::vector<TensorId> inputs,
                  std::vector<TensorId> outputs,
                  std::vector<Attribute> attributes = {});

  GraphBuilder& set_inputs(std::vector<TensorId> inputs);
  GraphBuilder& set_outputs(std::vector<TensorId> outputs);

  [[nodiscard]] Graph build() const;

 private:
  Graph graph_;
  TensorId next_tensor_id_ = 1;
  NodeId next_node_id_ = 1;
};

}  // namespace ir
}  // namespace uaii
