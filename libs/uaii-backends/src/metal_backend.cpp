#include "uaii/backends/metal_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

MetalBackend::MetalBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("metal", DeviceType::Metal, allocator, /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("Metal backend (host-fallback)");
  set_native_available(false);
  set_attention_host_fallback(true);
}

MetalBackend::~MetalBackend() { shutdown(); }

bool MetalBackend::native_compiled() noexcept { return native::metal_compiled(); }

Error MetalBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);
  set_attention_host_fallback(true);

  if (!force_host_fallback_ && native::metal_compiled()) {
    err = native::metal_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      // MatMul/Add/RMSNorm on-device when MSL pipelines built; Attention stays host.
      set_host_fallback(false);
      set_attention_host_fallback(true);
      set_details(std::string("Metal backend (native); ") +
                  native::metal_capability_details());
      return Error::success();
    }
    set_details("Metal backend (host-fallback; native init failed: " + err.message() + ")");
  } else if (force_host_fallback_) {
    set_details("Metal backend (host-fallback; forced)");
  } else {
    set_details("Metal backend (host-fallback; build with -DUAII_WITH_METAL=ON for native)");
  }
  return Error::success();
}

void MetalBackend::shutdown() noexcept {
  for (auto& kv : device_allocs_) {
    (void)native::metal_free(kv.first);
  }
  device_allocs_.clear();
  if (native_ready_) {
    native::metal_shutdown();
    native_ready_ = false;
  }
  HostExecutableBackend::shutdown();
}

Error MetalBackend::allocate(std::size_t bytes, void** out_ptr) {
  if (!native_ready_) {
    return HostExecutableBackend::allocate(bytes, out_ptr);
  }
  Error err = native::metal_allocate(bytes, out_ptr);
  if (!err.ok()) {
    return err;
  }
  device_allocs_.emplace_back(*out_ptr, bytes);
  return Error::success();
}

Error MetalBackend::free(void* ptr) noexcept {
  if (!native_ready_) {
    return HostExecutableBackend::free(ptr);
  }
  if (ptr == nullptr) {
    return Error::success();
  }
  for (auto it = device_allocs_.begin(); it != device_allocs_.end(); ++it) {
    if (it->first == ptr) {
      Error err = native::metal_free(ptr);
      device_allocs_.erase(it);
      return err;
    }
  }
  return Error::make(ErrorCode::NotFound, "pointer not owned by Metal backend");
}

Error MetalBackend::copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_h2d(host, device, bytes);
  }
  return native::metal_copy_h2d(host, device, bytes);
}

Error MetalBackend::copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2h(device, host, bytes);
  }
  return native::metal_copy_d2h(device, host, bytes);
}

Error MetalBackend::copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2d(src, dst, bytes);
  }
  return native::metal_copy_d2d(src, dst, bytes);
}

Error MetalBackend::synchronize() {
  if (!native_ready_) {
    return HostExecutableBackend::synchronize();
  }
  return native::metal_synchronize();
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
    return dispatch_via_host_staging(op_name, op_version, inputs, outputs, attrs);
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
