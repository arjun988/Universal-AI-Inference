#include "uaii/ir/registry.hpp"

namespace uaii {
namespace ir {

Error OperatorRegistry::register_schema(OpSchema schema) {
  if (schema.name.empty()) {
    return Error::make(ErrorCode::InvalidArgument, "op schema name is empty");
  }
  if (schema.version.empty()) {
    schema.version = "1";
  }
  const std::string key = op_key(schema.name, schema.version);
  schemas_[key] = std::move(schema);
  return Error::success();
}

const OpSchema* OperatorRegistry::find_schema(const std::string& name,
                                              const std::string& version) const {
  const auto it = schemas_.find(op_key(name, version));
  if (it == schemas_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<OpSchema> OperatorRegistry::list_schemas() const {
  std::vector<OpSchema> out;
  out.reserve(schemas_.size());
  for (const auto& kv : schemas_) {
    out.push_back(kv.second);
  }
  return out;
}

Error OperatorRegistry::register_operator(std::unique_ptr<IOperator> op) {
  if (!op) {
    return Error::make(ErrorCode::InvalidArgument, "null operator");
  }
  const auto sig = op->signature();
  const std::string key = op_key(sig.name, sig.version);
  operators_[key] = std::move(op);
  return Error::success();
}

IOperator* OperatorRegistry::find(const std::string& name,
                                  const std::string& version) const {
  const auto it = operators_.find(op_key(name, version));
  if (it == operators_.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::vector<OperatorSignature> OperatorRegistry::list() const {
  std::vector<OperatorSignature> out;
  out.reserve(operators_.size());
  for (const auto& kv : operators_) {
    out.push_back(kv.second->signature());
  }
  return out;
}

OperatorRegistry& default_registry() {
  static OperatorRegistry registry = [] {
    OperatorRegistry r;
    r.register_builtin_schemas();
    return r;
  }();
  return registry;
}

}  // namespace ir
}  // namespace uaii
