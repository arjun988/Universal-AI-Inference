#include "uaii/backends/metal_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

MetalBackend::MetalBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("metal", DeviceType::Metal, allocator, /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("Metal backend (host-fallback)");
  set_native_available(false);
}

bool MetalBackend::native_compiled() noexcept {
  return native::metal_compiled();
}

Error MetalBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);

  if (!force_host_fallback_ && native::metal_compiled()) {
    err = native::metal_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      set_host_fallback(false);
      set_details("Metal backend (native scaffold / device path)");
      return Error::success();
    }
    set_details("Metal backend (host-fallback; native init failed: " + err.message() +
                ")");
  } else if (force_host_fallback_) {
    set_details("Metal backend (forced host-fallback)");
  } else {
    set_details("Metal backend (host-fallback; build with -DUAII_WITH_METAL=ON for native)");
  }
  return Error::success();
}

Error MetalBackend::dispatch(const std::string& op_name,
                             const std::string& op_version,
                             const std::vector<kernels::TensorView>& inputs,
                             std::vector<kernels::TensorView>* outputs,
                             const std::vector<ir::Attribute>& attrs) {
  if (native_ready_) {
    Error err = native::metal_dispatch(op_name, op_version, inputs, outputs, attrs);
    if (err.ok()) {
      return err;
    }
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
