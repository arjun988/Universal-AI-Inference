#pragma once

#include "uaii/core/error.hpp"
#include "uaii/interfaces/types.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <cstddef>
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
  /// True when Attention (and related) still run on the host path.
  bool attention_host_fallback = false;
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

  [[nodiscard]] virtual Error copy_h2d(const void* host, void* device, std::size_t bytes) {
    (void)host;
    (void)device;
    (void)bytes;
    return Error::make(ErrorCode::NotImplemented, "copy_h2d not implemented");
  }

  /// Best-effort async host→device copy (CUDA copy stream when supported).
  [[nodiscard]] virtual Error copy_h2d_async(const void* host, void* device,
                                               std::size_t bytes) {
    (void)host;
    (void)device;
    (void)bytes;
    return Error::make(ErrorCode::NotImplemented, "copy_h2d_async not implemented");
  }

  [[nodiscard]] virtual Error copy_d2h(const void* device, void* host, std::size_t bytes) {
    (void)device;
    (void)host;
    (void)bytes;
    return Error::make(ErrorCode::NotImplemented, "copy_d2h not implemented");
  }

  [[nodiscard]] virtual Error copy_d2d(const void* src, void* dst, std::size_t bytes) {
    (void)src;
    (void)dst;
    (void)bytes;
    return Error::make(ErrorCode::NotImplemented, "copy_d2d not implemented");
  }

  [[nodiscard]] virtual Error synchronize() = 0;

  /// Execute one operator. Host-fallback GPU backends may run CPU kernels.
  [[nodiscard]] virtual Error dispatch(const std::string& op_name,
                                       const std::string& op_version,
                                       const std::vector<kernels::TensorView>& inputs,
                                       std::vector<kernels::TensorView>* outputs,
                                       const std::vector<ir::Attribute>& attrs) = 0;

  [[nodiscard]] virtual bool uses_host_fallback() const noexcept { return false; }

  /// True when Attention still executes via the host kernel path.
  [[nodiscard]] virtual bool attention_host_fallback() const noexcept { return false; }
};

}  // namespace uaii
