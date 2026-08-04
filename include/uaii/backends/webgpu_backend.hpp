#pragma once

#include "uaii/backends/host_executable.hpp"
#include "uaii/export.hpp"

namespace uaii {
namespace backends {

class UAII_API WebGpuBackend : public HostExecutableBackend {
 public:
  explicit WebGpuBackend(memory::Allocator* allocator = nullptr,
                         bool force_host_fallback = false);

  [[nodiscard]] Error initialize() override;
  [[nodiscard]] Error dispatch(const std::string& op_name,
                               const std::string& op_version,
                               const std::vector<kernels::TensorView>& inputs,
                               std::vector<kernels::TensorView>* outputs,
                               const std::vector<ir::Attribute>& attrs) override;

  [[nodiscard]] static bool native_compiled() noexcept;

 private:
  bool force_host_fallback_ = false;
  bool native_ready_ = false;
};

}  // namespace backends
}  // namespace uaii
