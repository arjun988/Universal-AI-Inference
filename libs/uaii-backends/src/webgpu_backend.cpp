#include "uaii/backends/webgpu_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

WebGpuBackend::WebGpuBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("webgpu", DeviceType::WebGpu, allocator,
                            /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("WebGPU backend (host-fallback)");
  set_native_available(false);
  set_attention_host_fallback(true);
}

WebGpuBackend::~WebGpuBackend() { shutdown(); }

bool WebGpuBackend::native_compiled() noexcept { return native::webgpu_compiled(); }

Error WebGpuBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);
  set_attention_host_fallback(true);

  if (!force_host_fallback_ && native::webgpu_compiled()) {
    err = native::webgpu_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      set_host_fallback(true);
      set_attention_host_fallback(true);
      set_details(std::string("WebGPU backend (native buffers); ") +
                  native::webgpu_capability_details());
      return Error::success();
    }
    set_details("WebGPU backend (host-fallback; native init failed: " + err.message() +
                ")");
  } else if (force_host_fallback_) {
    set_details("WebGPU backend (host-fallback; forced)");
  } else {
    set_details(
        "WebGPU backend (host-fallback; build with -DUAII_WITH_WEBGPU=ON for native)");
  }
  return Error::success();
}

void WebGpuBackend::shutdown() noexcept {
  for (auto& kv : device_allocs_) {
    (void)native::webgpu_free(kv.first);
  }
  device_allocs_.clear();
  if (native_ready_) {
    native::webgpu_shutdown();
    native_ready_ = false;
  }
  HostExecutableBackend::shutdown();
}

Error WebGpuBackend::allocate(std::size_t bytes, void** out_ptr) {
  if (!native_ready_) {
    return HostExecutableBackend::allocate(bytes, out_ptr);
  }
  Error err = native::webgpu_allocate(bytes, out_ptr);
  if (!err.ok()) {
    return err;
  }
  device_allocs_.emplace_back(*out_ptr, bytes);
  return Error::success();
}

Error WebGpuBackend::free(void* ptr) noexcept {
  if (!native_ready_) {
    return HostExecutableBackend::free(ptr);
  }
  if (ptr == nullptr) {
    return Error::success();
  }
  for (auto it = device_allocs_.begin(); it != device_allocs_.end(); ++it) {
    if (it->first == ptr) {
      Error err = native::webgpu_free(ptr);
      device_allocs_.erase(it);
      return err;
    }
  }
  return Error::make(ErrorCode::NotFound, "pointer not owned by WebGPU backend");
}

Error WebGpuBackend::copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_h2d(host, device, bytes);
  }
  return native::webgpu_copy_h2d(host, device, bytes);
}

Error WebGpuBackend::copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2h(device, host, bytes);
  }
  return native::webgpu_copy_d2h(device, host, bytes);
}

Error WebGpuBackend::copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2d(src, dst, bytes);
  }
  return native::webgpu_copy_d2d(src, dst, bytes);
}

Error WebGpuBackend::synchronize() {
  if (!native_ready_) {
    return HostExecutableBackend::synchronize();
  }
  return native::webgpu_synchronize();
}

Error WebGpuBackend::dispatch(const std::string& op_name,
                              const std::string& op_version,
                              const std::vector<kernels::TensorView>& inputs,
                              std::vector<kernels::TensorView>* outputs,
                              const std::vector<ir::Attribute>& attrs) {
  if (native_ready_) {
    Error err = native::webgpu_dispatch(op_name, op_version, inputs, outputs, attrs);
    if (err.ok()) {
      return err;
    }
    return dispatch_via_host_staging(op_name, op_version, inputs, outputs, attrs);
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
