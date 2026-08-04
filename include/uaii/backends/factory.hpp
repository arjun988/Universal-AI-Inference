#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/backend.hpp"
#include "uaii/memory/allocator.hpp"

#include <memory>
#include <string>
#include <vector>

namespace uaii {
namespace backends {

struct BackendCreateOptions {
  memory::Allocator* allocator = nullptr;
  /// Prefer vendor native path when compiled in; otherwise host-fallback.
  bool prefer_native = true;
  /// Force host-fallback even if native is compiled (useful for parity).
  bool force_host_fallback = false;
};

struct BackendDescriptor {
  std::string name;
  DeviceType device_type = DeviceType::Unknown;
  bool always_available = true;   // host path always works
  bool native_compiled = false;   // built with UAII_WITH_* 
  std::string description;
};

[[nodiscard]] UAII_API std::vector<BackendDescriptor> list_backends();

[[nodiscard]] UAII_API bool backend_exists(const std::string& name) noexcept;

/// Create a backend by name: cpu, cuda, metal, vulkan, webgpu, rocm.
[[nodiscard]] UAII_API Error create_backend(const std::string& name,
                                            const BackendCreateOptions& options,
                                            std::unique_ptr<IBackend>* out);

[[nodiscard]] UAII_API const char* device_type_backend_name(DeviceType type) noexcept;

}  // namespace backends
}  // namespace uaii
