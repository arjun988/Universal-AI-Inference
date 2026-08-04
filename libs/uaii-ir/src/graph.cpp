#include "uaii/ir/graph.hpp"

namespace uaii {
namespace ir {

const Tensor* Graph::find_tensor(TensorId id) const {
  for (const auto& t : tensors) {
    if (t.id == id) {
      return &t;
    }
  }
  return nullptr;
}

Tensor* Graph::find_tensor(TensorId id) {
  for (auto& t : tensors) {
    if (t.id == id) {
      return &t;
    }
  }
  return nullptr;
}

const Node* Graph::find_node(NodeId id) const {
  for (const auto& n : nodes) {
    if (n.id == id) {
      return &n;
    }
  }
  return nullptr;
}

void Graph::clear() {
  *this = Graph{};
}

GraphBuilder::GraphBuilder(std::string name) {
  graph_.name = std::move(name);
  graph_.producer = "uaii-graph-builder";
  graph_.version = kCurrentIrVersion;
}

GraphBuilder& GraphBuilder::set_producer(std::string producer) {
  graph_.producer = std::move(producer);
  return *this;
}

GraphBuilder& GraphBuilder::set_domain(std::string domain) {
  graph_.domain = std::move(domain);
  return *this;
}

GraphBuilder& GraphBuilder::set_version(IrVersion version) {
  graph_.version = version;
  return *this;
}

GraphBuilder& GraphBuilder::set_metadata(std::string key, std::string value) {
  graph_.metadata[std::move(key)] = std::move(value);
  return *this;
}

TensorId GraphBuilder::add_tensor(std::string name, DType dtype, Shape shape,
                                  StorageHint hint) {
  Tensor t;
  t.id = next_tensor_id_++;
  t.name = std::move(name);
  t.dtype = dtype;
  t.shape = std::move(shape);
  t.storage_hint = hint;
  graph_.tensors.push_back(std::move(t));
  return graph_.tensors.back().id;
}

TensorId GraphBuilder::add_weight(std::string name, DType dtype, Shape shape,
                                  std::string weight_ref) {
  Tensor t;
  t.id = next_tensor_id_++;
  t.name = std::move(name);
  t.dtype = dtype;
  t.shape = std::move(shape);
  t.is_weight = true;
  t.storage_hint = StorageHint::External;
  t.weight_ref = std::move(weight_ref);
  graph_.tensors.push_back(std::move(t));
  return graph_.tensors.back().id;
}

NodeId GraphBuilder::add_node(std::string name,
                              std::string op_name,
                              std::string op_version,
                              std::vector<TensorId> inputs,
                              std::vector<TensorId> outputs,
                              std::vector<Attribute> attributes) {
  Node n;
  n.id = next_node_id_++;
  n.name = std::move(name);
  n.op_name = std::move(op_name);
  n.op_version = std::move(op_version);
  n.inputs = std::move(inputs);
  n.outputs = std::move(outputs);
  n.attributes = std::move(attributes);
  graph_.nodes.push_back(std::move(n));
  return graph_.nodes.back().id;
}

GraphBuilder& GraphBuilder::set_inputs(std::vector<TensorId> inputs) {
  graph_.inputs = std::move(inputs);
  return *this;
}

GraphBuilder& GraphBuilder::set_outputs(std::vector<TensorId> outputs) {
  graph_.outputs = std::move(outputs);
  return *this;
}

Graph GraphBuilder::build() const {
  return graph_;
}

}  // namespace ir
}  // namespace uaii
