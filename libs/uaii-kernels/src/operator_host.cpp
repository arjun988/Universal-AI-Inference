#include "uaii/plugins/operator_host.hpp"

namespace uaii {
namespace plugins {

OperatorHostRegistry& OperatorHostRegistry::instance() {
  static OperatorHostRegistry reg;
  return reg;
}

void OperatorHostRegistry::register_op(std::string name, HostOpFn fn) {
  std::lock_guard<std::mutex> lock(mu_);
  ops_[std::move(name)] = std::move(fn);
}

bool OperatorHostRegistry::has(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mu_);
  return ops_.count(name) != 0;
}

Error OperatorHostRegistry::dispatch(const std::string& name,
                                     const std::vector<kernels::TensorView>& inputs,
                                     std::vector<kernels::TensorView>* outputs,
                                     const std::vector<ir::Attribute>& attrs) const {
  HostOpFn fn;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = ops_.find(name);
    if (it == ops_.end()) {
      return Error::make(ErrorCode::NotFound, "plugin op not registered: " + name);
    }
    fn = it->second;
  }
  return fn(inputs, outputs, attrs);
}

std::vector<std::string> OperatorHostRegistry::names() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> out;
  out.reserve(ops_.size());
  for (const auto& kv : ops_) out.push_back(kv.first);
  return out;
}

}  // namespace plugins
}  // namespace uaii
