#include "uaii/backends/rocm_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

RocmBackend::RocmBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("rocm", DeviceType::Rocm, allocator, /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("ROCm backend (host-fallback)");
  set_native_available(false);
  set_attention_host_fallback(true);
}

RocmBackend::~RocmBackend() { shutdown(); }

bool RocmBackend::native_compiled() noexcept { return native::rocm_compiled(); }

Error RocmBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);
  set_attention_host_fallback(true);

  if (!force_host_fallback_ && native::rocm_compiled()) {
    err = native::rocm_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      set_host_fallback(false);
      set_attention_host_fallback(true);
      set_details(std::string("ROCm backend (native); ") + native::rocm_capability_details());
      return Error::success();
    }
    set_details("ROCm backend (host-fallback; native init failed: " + err.message() + ")");
  } else if (force_host_fallback_) {
    set_details("ROCm backend (host-fallback; forced)");
  } else {
    set_details("ROCm backend (host-fallback; build with -DUAII_WITH_ROCM=ON for native)");
  }
  return Error::success();
}

void RocmBackend::shutdown() noexcept {
  for (auto& kv : device_allocs_) {
    (void)native::rocm_free(kv.first);
  }
  device_allocs_.clear();
  if (native_ready_) {
    native::rocm_shutdown();
    native_ready_ = false;
  }
  HostExecutableBackend::shutdown();
}

Error RocmBackend::allocate(std::size_t bytes, void** out_ptr) {
  if (!native_ready_) {
    return HostExecutableBackend::allocate(bytes, out_ptr);
  }
  Error err = native::rocm_allocate(bytes, out_ptr);
  if (!err.ok()) {
    return err;
  }
  device_allocs_.emplace_back(*out_ptr, bytes);
  return Error::success();
}

Error RocmBackend::free(void* ptr) noexcept {
  if (!native_ready_) {
    return HostExecutableBackend::free(ptr);
  }
  if (ptr == nullptr) {
    return Error::success();
  }
  for (auto it = device_allocs_.begin(); it != device_allocs_.end(); ++it) {
    if (it->first == ptr) {
      Error err = native::rocm_free(ptr);
      device_allocs_.erase(it);
      return err;
    }
  }
  return Error::make(ErrorCode::NotFound, "pointer not owned by ROCm backend");
}

Error RocmBackend::copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_h2d(host, device, bytes);
  }
  return native::rocm_copy_h2d(host, device, bytes);
}

Error RocmBackend::copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2h(device, host, bytes);
  }
  return native::rocm_copy_d2h(device, host, bytes);
}

Error RocmBackend::copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2d(src, dst, bytes);
  }
  return native::rocm_copy_d2d(src, dst, bytes);
}

Error RocmBackend::synchronize() {
  if (!native_ready_) {
    return HostExecutableBackend::synchronize();
  }
  return native::rocm_synchronize();
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
    return dispatch_via_host_staging(op_name, op_version, inputs, outputs, attrs);
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
