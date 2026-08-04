#include "uaii/backends/cpu_backend.hpp"

namespace uaii {
namespace backends {

CpuBackend::CpuBackend(memory::Allocator* allocator)
    : HostExecutableBackend("cpu", DeviceType::Cpu, allocator, /*host_fallback=*/false) {
  set_details("Phase 3/5 host CPU backend (f32 kernels)");
  set_native_available(true);
}

}  // namespace backends
}  // namespace uaii
