#pragma once

#include "uaii/core/error.hpp"

#include <map>
#include <string>
#include <vector>

namespace uaii {
namespace ir {
namespace json {

struct Value;

using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
  // Names avoid shadowing the Array/Object type aliases above.
  enum class Kind { Null, Bool, Number, String, Arr, Obj };

  Kind kind = Kind::Null;
  bool b = false;
  double number = 0.0;
  std::string str;
  Array arr;
  Object obj;

  [[nodiscard]] static Value null() { return Value{}; }
  [[nodiscard]] static Value boolean(bool v) {
    Value x;
    x.kind = Kind::Bool;
    x.b = v;
    return x;
  }
  [[nodiscard]] static Value num(double v) {
    Value x;
    x.kind = Kind::Number;
    x.number = v;
    return x;
  }
  [[nodiscard]] static Value string(std::string v) {
    Value x;
    x.kind = Kind::String;
    x.str = std::move(v);
    return x;
  }
  [[nodiscard]] static Value array(Array v) {
    Value x;
    x.kind = Kind::Arr;
    x.arr = std::move(v);
    return x;
  }
  [[nodiscard]] static Value object(Object v) {
    Value x;
    x.kind = Kind::Obj;
    x.obj = std::move(v);
    return x;
  }

  [[nodiscard]] bool is_null() const { return kind == Kind::Null; }
  [[nodiscard]] bool is_bool() const { return kind == Kind::Bool; }
  [[nodiscard]] bool is_number() const { return kind == Kind::Number; }
  [[nodiscard]] bool is_string() const { return kind == Kind::String; }
  [[nodiscard]] bool is_array() const { return kind == Kind::Arr; }
  [[nodiscard]] bool is_object() const { return kind == Kind::Obj; }
};

[[nodiscard]] Error parse(const std::string& text, Value* out);
[[nodiscard]] std::string stringify(const Value& value, bool pretty = true);

[[nodiscard]] const Value* get(const Object& obj, const char* key);
[[nodiscard]] Error require_object(const Value& v, const Object** out, const char* ctx);
[[nodiscard]] Error require_array(const Value& v, const Array** out, const char* ctx);

}  // namespace json
}  // namespace ir
}  // namespace uaii
