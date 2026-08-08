#include "uaii/backends/cuda_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

CudaBackend::CudaBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("cuda", DeviceType::Cuda, allocator, /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("CUDA backend (host-fallback)");
  set_native_available(false);
  set_attention_host_fallback(true);
}

CudaBackend::~CudaBackend() {
  shutdown();
}

bool CudaBackend::native_compiled() noexcept {
  return native::cuda_compiled();
}

bool CudaBackend::pointer_on_device(const void* p) const noexcept {
  return native_ready_ && native::cuda_is_device_ptr(p);
}

bool CudaBackend::native_device_available() noexcept {
  if (!native::cuda_compiled()) {
    return false;
  }
  return native::cuda_probe_device();
}

Error CudaBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);
  set_attention_host_fallback(true);

  if (!force_host_fallback_ && native::cuda_compiled()) {
    err = native::cuda_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      // MatMul + Add + RMSNorm on device when tensors are device-resident.
      // Session still allocates host RAM today — dispatch falls back to CPU kernels
      // in place until device upload is wired (avoids CUDA D2H invalid argument).
      set_host_fallback(false);
      set_details(std::string("CUDA backend (native); ") + native::cuda_capability_details() +
                  " | note: host-resident session tensors use CPU kernels until device upload");
      return Error::success();
    }
    set_details("CUDA backend (host-fallback; native init failed: " + err.message() + ")");
  } else if (force_host_fallback_) {
    set_details("CUDA backend (host-fallback; forced)");
  } else {
    set_details("CUDA backend (host-fallback; build with -DUAII_WITH_CUDA=ON for native)");
  }
  return Error::success();
}

void CudaBackend::shutdown() noexcept {
  for (auto& kv : device_allocs_) {
    (void)native::cuda_free(kv.first);
  }
  device_allocs_.clear();
  if (native_ready_) {
    native::cuda_shutdown();
    native_ready_ = false;
  }
  HostExecutableBackend::shutdown();
}

BackendCapabilities CudaBackend::capabilities() const {
  BackendCapabilities caps = HostExecutableBackend::capabilities();
  if (native_ready_) {
    caps.supports_async = true;
  }
  return caps;
}

Error CudaBackend::allocate(std::size_t bytes, void** out_ptr) {
  if (!native_ready_) {
    return HostExecutableBackend::allocate(bytes, out_ptr);
  }
  Error err = native::cuda_allocate(bytes, out_ptr);
  if (!err.ok()) {
    return err;
  }
  device_allocs_.emplace_back(*out_ptr, bytes);
  return Error::success();
}

Error CudaBackend::free(void* ptr) noexcept {
  if (!native_ready_) {
    return HostExecutableBackend::free(ptr);
  }
  if (ptr == nullptr) {
    return Error::success();
  }
  for (auto it = device_allocs_.begin(); it != device_allocs_.end(); ++it) {
    if (it->first == ptr) {
      Error err = native::cuda_free(ptr);
      device_allocs_.erase(it);
      return err;
    }
  }
  return Error::make(ErrorCode::NotFound, "pointer not owned by CUDA backend");
}

Error CudaBackend::copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_h2d(host, device, bytes);
  }
  // Session may still pass host destinations before device upload exists.
  if (!native::cuda_is_device_ptr(device)) {
    return HostExecutableBackend::copy_h2d(host, device, bytes);
  }
  return native::cuda_copy_h2d(host, device, bytes);
}

Error CudaBackend::copy_h2d_async(const void* host, void* device, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_h2d_async(host, device, bytes);
  }
  if (!native::cuda_is_device_ptr(device)) {
    return HostExecutableBackend::copy_h2d_async(host, device, bytes);
  }
  return native::cuda_copy_h2d_async(host, device, bytes);
}

Error CudaBackend::copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2h(device, host, bytes);
  }
  // Host-resident tensors must not be passed to cudaMemcpy DeviceToHost.
  if (!native::cuda_is_device_ptr(device)) {
    return HostExecutableBackend::copy_d2h(device, host, bytes);
  }
  return native::cuda_copy_d2h(device, host, bytes);
}

Error CudaBackend::copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (!native_ready_) {
    return HostExecutableBackend::copy_d2d(src, dst, bytes);
  }
  if (!native::cuda_is_device_ptr(src) || !native::cuda_is_device_ptr(dst)) {
    return HostExecutableBackend::copy_d2d(src, dst, bytes);
  }
  return native::cuda_copy_d2d(src, dst, bytes);
}

Error CudaBackend::synchronize() {
  if (!native_ready_) {
    return HostExecutableBackend::synchronize();
  }
  return native::cuda_synchronize();
}

Error CudaBackend::dispatch(const std::string& op_name,
                            const std::string& op_version,
                            const std::vector<kernels::TensorView>& inputs,
                            std::vector<kernels::TensorView>* outputs,
                            const std::vector<ir::Attribute>& attrs) {
  if (native_ready_) {
    auto on_device = [](const kernels::TensorView& t) {
      return t.data != nullptr && t.nbytes > 0 && native::cuda_is_device_ptr(t.data);
    };
    bool any_host = false;
    bool any_device = false;
    for (const auto& t : inputs) {
      if (t.data == nullptr || t.nbytes == 0) continue;
      if (on_device(t)) any_device = true;
      else any_host = true;
    }
    if (outputs) {
      for (const auto& t : *outputs) {
        if (t.data == nullptr || t.nbytes == 0) continue;
        if (on_device(t)) any_device = true;
        else any_host = true;
      }
    }

    // Host-resident session buffers: run CPU kernels in place (do not cudaMemcpy).
    if (!any_device) {
      return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
    }
    if (any_host && any_device) {
      // Mixed residency — stage everything through host.
      return dispatch_via_host_staging(op_name, op_version, inputs, outputs, attrs);
    }

    Error err = native::cuda_dispatch(op_name, op_version, inputs, outputs, attrs);
    if (err.ok()) {
      return err;
    }
    // Device buffers: stage through host for unimplemented ops (Attention, etc.).
    return dispatch_via_host_staging(op_name, op_version, inputs, outputs, attrs);
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
