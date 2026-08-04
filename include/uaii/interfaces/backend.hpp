#pragma once

#include "uaii/core/error.hpp"
#include "uaii/interfaces/types.hpp"

#include <string>
#include <vector>

namespace uaii {

/// Capability snapshot returned by a backend (Phase 1 contract).
struct BackendCapabilities {
  std::string name;
  DeviceType device_type = DeviceType::Unknown;
  std::vector<DType> supported_dtypes;
  bool supports_profiling = false;
  bool supports_async = false;
  std::string details;
};

/// Hardware abstraction interface. Implementations arrive in later phases /
/// as plugins. Phase 1 defines the contract only.
class IBackend {
 public:
  virtual ~IBackend() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual DeviceType device_type() const = 0;

  [[nodiscard]] virtual Error initialize() = 0;
  virtual void shutdown() noexcept = 0;

  [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;

  /// Placeholder allocation API — fleshed out with memory handles in Phase 3.
  [[nodiscard]] virtual Error allocate(std::size_t bytes, void** out_ptr) = 0;
  [[nodiscard]] virtual Error free(void* ptr) noexcept = 0;

  [[nodiscard]] virtual Error synchronize() = 0;
};

}  // namespace uaii
