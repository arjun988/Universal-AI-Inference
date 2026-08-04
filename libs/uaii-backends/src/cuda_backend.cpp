#include "uaii/backends/cuda_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

CudaBackend::CudaBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("cuda", DeviceType::Cuda, allocator, /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("CUDA backend (host-fallback)");
  set_native_available(false);
}

bool CudaBackend::native_compiled() noexcept {
  return native::cuda_compiled();
}

Error CudaBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);

  if (!force_host_fallback_ && native::cuda_compiled()) {
    err = native::cuda_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      set_host_fallback(false);
      set_details("CUDA backend (native scaffold / device path)");
      return Error::ok();
    }
    set_details("CUDA backend (host-fallback; native init failed: " + err.message() +
                ")");
  } else if (force_host_fallback_) {
    set_details("CUDA backend (forced host-fallback)");
  } else {
    set_details("CUDA backend (host-fallback; build with -DUAII_WITH_CUDA=ON for native)");
  }
  return Error::ok();
}

Error CudaBackend::dispatch(const std::string& op_name,
                            const std::string& op_version,
                            const std::vector<kernels::TensorView>& inputs,
                            std::vector<kernels::TensorView>* outputs,
                            const std::vector<ir::Attribute>& attrs) {
  if (native_ready_) {
    Error err = native::cuda_dispatch(op_name, op_version, inputs, outputs, attrs);
    if (err.ok()) {
      return err;
    }
    // Fall through to host kernels if native op missing.
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
