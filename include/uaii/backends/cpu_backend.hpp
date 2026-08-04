#pragma once

#include "uaii/backends/host_executable.hpp"
#include "uaii/export.hpp"

namespace uaii {
namespace backends {

/// CPU device backend: host memory + kernel dispatch.
class UAII_API CpuBackend : public HostExecutableBackend {
 public:
  explicit CpuBackend(memory::Allocator* allocator = nullptr);
};

}  // namespace backends
}  // namespace uaii
