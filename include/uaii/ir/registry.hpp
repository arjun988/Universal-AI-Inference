#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/operator_iface.hpp"
#include "uaii/ir/op.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace ir {

/// Dynamic operator registry: schemas (validation) + optional executables (Phase 3+).
class UAII_API OperatorRegistry : public IOperatorRegistry {
 public:
  OperatorRegistry() = default;

  /// Register a schema used by the graph validator.
  [[nodiscard]] Error register_schema(OpSchema schema);

  [[nodiscard]] const OpSchema* find_schema(const std::string& name,
                                            const std::string& version) const;

  [[nodiscard]] std::vector<OpSchema> list_schemas() const;

  /// IOperatorRegistry — executable kernels (optional in Phase 2).
  [[nodiscard]] Error register_operator(std::unique_ptr<IOperator> op) override;
  [[nodiscard]] IOperator* find(const std::string& name,
                                const std::string& version) const override;
  [[nodiscard]] std::vector<OperatorSignature> list() const override;

  /// Install the built-in Phase 2 operator schemas (MatMul, Softmax, …).
  void register_builtin_schemas();

  [[nodiscard]] std::size_t schema_count() const noexcept { return schemas_.size(); }

 private:
  std::unordered_map<std::string, OpSchema> schemas_;
  std::unordered_map<std::string, std::unique_ptr<IOperator>> operators_;
};

/// Process-wide default registry (lazy builtins).
[[nodiscard]] UAII_API OperatorRegistry& default_registry();

}  // namespace ir
}  // namespace uaii
