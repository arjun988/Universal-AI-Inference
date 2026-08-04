#include "uaii/backends/rocm_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

RocmBackend::RocmBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("rocm", DeviceType::Rocm, allocator, /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("ROCm backend (host-fallback)");
  set_native_available(false);
}

bool RocmBackend::native_compiled() noexcept {
  return native::rocm_compiled();
}

Error RocmBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);

  if (!force_host_fallback_ && native::rocm_compiled()) {
    err = native::rocm_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      set_host_fallback(false);
      set_details("ROCm backend (native scaffold / device path)");
      return Error::ok();
    }
    set_details("ROCm backend (host-fallback; native init failed: " + err.message() +
                ")");
  } else if (force_host_fallback_) {
    set_details("ROCm backend (forced host-fallback)");
  } else {
    set_details("ROCm backend (host-fallback; build with -DUAII_WITH_ROCM=ON for native)");
  }
  return Error::ok();
}

Error RocmBackend::dispatch(const std::string& op_name,
                            const std::string& op_version,
                            const std::vector<kernels::TensorView>& inputs,
                            std::vector<kernels::TensorView>* outputs,
                            const std::vector<ir::Attribute>& attrs) {
  if (native_ready_) {
    Error err = native::rocm_dispatch(op_name, op_version, inputs, outputs, attrs);
    if (err.ok()) {
      return err;
    }
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
