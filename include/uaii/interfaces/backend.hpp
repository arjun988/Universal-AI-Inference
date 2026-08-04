#pragma once

#include "uaii/core/error.hpp"
#include "uaii/interfaces/types.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <string>
#include <vector>

namespace uaii {

/// Capability snapshot returned by a backend.
struct BackendCapabilities {
  std::string name;
  DeviceType device_type = DeviceType::Unknown;
  std::vector<DType> supported_dtypes;
  bool supports_profiling = false;
  bool supports_async = false;
  bool host_fallback = false;
  bool native_available = false;
  std::string details;
};

/// Hardware abstraction interface (Phase 5: all backends are executable).
class IBackend {
 public:
  virtual ~IBackend() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual DeviceType device_type() const = 0;

  [[nodiscard]] virtual Error initialize() = 0;
  virtual void shutdown() noexcept = 0;

  [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;

  [[nodiscard]] virtual Error allocate(std::size_t bytes, void** out_ptr) = 0;
  [[nodiscard]] virtual Error free(void* ptr) noexcept = 0;

  [[nodiscard]] virtual Error synchronize() = 0;

  /// Execute one operator. Host-fallback GPU backends may run CPU kernels.
  [[nodiscard]] virtual Error dispatch(const std::string& op_name,
                                       const std::string& op_version,
                                       const std::vector<kernels::TensorView>& inputs,
                                       std::vector<kernels::TensorView>* outputs,
                                       const std::vector<ir::Attribute>& attrs) = 0;

  [[nodiscard]] virtual bool uses_host_fallback() const noexcept { return false; }
};

}  // namespace uaii
