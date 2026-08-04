#pragma once

#include "uaii/interfaces/types.hpp"
#include "uaii/ir/attribute.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace uaii {
namespace ir {

/// Schema describing an operator family for validation / planning.
struct OpSchema {
  std::string name;
  std::string version = "1";
  /// -1 means variadic.
  int min_inputs = 0;
  int max_inputs = 0;
  int min_outputs = 1;
  int max_outputs = 1;
  std::vector<std::string> known_attributes;
  std::string description;
};

/// Operator invocation inside a graph node (not the executable kernel).
struct OpSpec {
  std::string name;
  std::string version = "1";
  std::vector<Attribute> attributes;
};

[[nodiscard]] inline std::string op_key(const std::string& name, const std::string& version) {
  return name + "@" + version;
}

}  // namespace ir
}  // namespace uaii
