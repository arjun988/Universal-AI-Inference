#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace plugins {

using HostOpFn = std::function<Error(const std::vector<kernels::TensorView>&,
                                     std::vector<kernels::TensorView>*,
                                     const std::vector<ir::Attribute>&)>;

/// Runtime-extensible operator table consulted by CPU dispatch before builtins.
class UAII_API OperatorHostRegistry {
 public:
  static OperatorHostRegistry& instance();

  void register_op(std::string name, HostOpFn fn);
  [[nodiscard]] bool has(const std::string& name) const;
  [[nodiscard]] Error dispatch(const std::string& name,
                               const std::vector<kernels::TensorView>& inputs,
                               std::vector<kernels::TensorView>* outputs,
                               const std::vector<ir::Attribute>& attrs) const;
  [[nodiscard]] std::vector<std::string> names() const;

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, HostOpFn> ops_;
};

}  // namespace plugins
}  // namespace uaii
