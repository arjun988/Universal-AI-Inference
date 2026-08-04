#pragma once

#include "uaii/interfaces/types.hpp"
#include "uaii/ir/attribute.hpp"

#include <string>
#include <vector>

namespace uaii {
namespace ir {

struct Node {
  NodeId id = 0;
  std::string name;
  std::string op_name;
  std::string op_version = "1";
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::vector<Attribute> attributes;
};

}  // namespace ir
}  // namespace uaii
