#pragma once

#include "uaii/core/error.hpp"
#include "uaii/interfaces/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace uaii {

struct OperatorAttribute {
  std::string key;
  std::string value;
};

struct OperatorSignature {
  std::string name;
  std::string version;
  std::vector<DType> input_dtypes;
  std::vector<DType> output_dtypes;
};

/// Executable operator / kernel entry. Registered dynamically via plugins.
class IOperator {
 public:
  virtual ~IOperator() = default;

  [[nodiscard]] virtual OperatorSignature signature() const = 0;

  /// Execute against bound tensors (Phase 3+). Phase 1 defines the contract.
  [[nodiscard]] virtual Error execute() = 0;
};

/// Dynamic operator registry interface (implemented in Phase 2).
class IOperatorRegistry {
 public:
  virtual ~IOperatorRegistry() = default;

  [[nodiscard]] virtual Error register_operator(std::unique_ptr<IOperator> op) = 0;
  [[nodiscard]] virtual IOperator* find(const std::string& name,
                                        const std::string& version) const = 0;
  [[nodiscard]] virtual std::vector<OperatorSignature> list() const = 0;
};

}  // namespace uaii
