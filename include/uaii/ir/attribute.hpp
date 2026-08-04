#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace uaii {
namespace ir {

enum class AttributeType {
  String = 0,
  Int,
  Float,
  Bool,
  IntArray,
  FloatArray,
};

using AttributeValue = std::variant<std::string,
                                    std::int64_t,
                                    double,
                                    bool,
                                    std::vector<std::int64_t>,
                                    std::vector<double>>;

struct Attribute {
  std::string key;
  AttributeType type = AttributeType::String;
  AttributeValue value = std::string{};
};

[[nodiscard]] inline Attribute make_string_attr(std::string key, std::string value) {
  Attribute a;
  a.key = std::move(key);
  a.type = AttributeType::String;
  a.value = std::move(value);
  return a;
}

[[nodiscard]] inline Attribute make_int_attr(std::string key, std::int64_t value) {
  Attribute a;
  a.key = std::move(key);
  a.type = AttributeType::Int;
  a.value = value;
  return a;
}

[[nodiscard]] inline Attribute make_float_attr(std::string key, double value) {
  Attribute a;
  a.key = std::move(key);
  a.type = AttributeType::Float;
  a.value = value;
  return a;
}

[[nodiscard]] inline Attribute make_bool_attr(std::string key, bool value) {
  Attribute a;
  a.key = std::move(key);
  a.type = AttributeType::Bool;
  a.value = value;
  return a;
}

[[nodiscard]] inline const char* to_string(AttributeType type) noexcept {
  switch (type) {
    case AttributeType::String: return "string";
    case AttributeType::Int: return "int";
    case AttributeType::Float: return "float";
    case AttributeType::Bool: return "bool";
    case AttributeType::IntArray: return "int_array";
    case AttributeType::FloatArray: return "float_array";
  }
  return "unknown";
}

}  // namespace ir
}  // namespace uaii
