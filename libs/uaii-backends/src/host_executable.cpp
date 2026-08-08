#include "uaii/backends/host_executable.hpp"

#include "uaii/kernels/kernels.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace uaii {
namespace backends {

HostExecutableBackend::HostExecutableBackend(std::string name,
                                             DeviceType device_type,
                                             memory::Allocator* allocator,
                                             bool host_fallback)
    : name_(std::move(name)),
      device_type_(device_type),
      host_fallback_(host_fallback),
      allocator_(allocator) {
  if (allocator_ == nullptr) {
    owned_allocator_ = std::make_unique<memory::Allocator>();
    allocator_ = owned_allocator_.get();
  }
  if (details_.empty()) {
    details_ = host_fallback_ ? "host-fallback executable backend"
                              : "host executable backend";
  }
}

HostExecutableBackend::~HostExecutableBackend() {
  shutdown();
}

Error HostExecutableBackend::initialize() {
  initialized_ = true;
  return Error::success();
}

void HostExecutableBackend::shutdown() noexcept {
  for (auto& kv : orphan_allocs_) {
    allocator_->deallocate_bytes(kv.first, kv.second);
  }
  orphan_allocs_.clear();
  initialized_ = false;
}

BackendCapabilities HostExecutableBackend::capabilities() const {
  BackendCapabilities caps;
  caps.name = name_;
  caps.device_type = device_type_;
  caps.supported_dtypes = {DType::F32};
  caps.supports_profiling = false;
  caps.supports_async = false;
  caps.host_fallback = host_fallback_;
  caps.native_available = native_available_;
  caps.attention_host_fallback = attention_host_fallback_;
  caps.details = details_;
  return caps;
}

Error HostExecutableBackend::allocate(std::size_t bytes, void** out_ptr) {
  Error err = allocator_->allocate_bytes(bytes, out_ptr);
  if (!err.ok()) {
    return err;
  }
  orphan_allocs_.emplace_back(*out_ptr, bytes);
  return Error::success();
}

Error HostExecutableBackend::free(void* ptr) noexcept {
  if (ptr == nullptr) {
    return Error::success();
  }
  for (auto it = orphan_allocs_.begin(); it != orphan_allocs_.end(); ++it) {
    if (it->first == ptr) {
      allocator_->deallocate_bytes(it->first, it->second);
      orphan_allocs_.erase(it);
      return Error::success();
    }
  }
  return Error::make(ErrorCode::NotFound, "pointer not owned by backend");
}

Error HostExecutableBackend::copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (bytes == 0) {
    return Error::success();
  }
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "copy_h2d null pointer");
  }
  std::memcpy(device, host, bytes);
  return Error::success();
}

Error HostExecutableBackend::copy_h2d_async(const void* host, void* device,
                                            std::size_t bytes) {
  return copy_h2d(host, device, bytes);
}

Error HostExecutableBackend::copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (bytes == 0) {
    return Error::success();
  }
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "copy_d2h null pointer");
  }
  std::memcpy(host, device, bytes);
  return Error::success();
}

Error HostExecutableBackend::copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (bytes == 0) {
    return Error::success();
  }
  if (src == nullptr || dst == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "copy_d2d null pointer");
  }
  std::memcpy(dst, src, bytes);
  return Error::success();
}

Error HostExecutableBackend::synchronize() {
  return Error::success();
}

Error HostExecutableBackend::dispatch(const std::string& op_name,
                                      const std::string& op_version,
                                      const std::vector<kernels::TensorView>& inputs,
                                      std::vector<kernels::TensorView>* outputs,
                                      const std::vector<ir::Attribute>& attrs) {
  if (!initialized_) {
    return Error::make(ErrorCode::InvalidArgument, name_ + " backend not initialized");
  }
  return kernels::dispatch_cpu(op_name, op_version, inputs, outputs, attrs);
}

Error HostExecutableBackend::dispatch_on_host_path(
    const std::string& op_name,
    const std::string& op_version,
    const std::vector<kernels::TensorView>& inputs,
    std::vector<kernels::TensorView>* outputs,
    const std::vector<ir::Attribute>& attrs) {
  // Only stage through D2H/H2D when tensors actually live on the device.
  // Session currently allocates host RAM even for CUDA — staging host pointers
  // as device memory yields cudaMemcpy "invalid argument".
  bool need_stage = false;
  if (device_type_ != DeviceType::Cpu && native_available_) {
    for (const auto& t : inputs) {
      if (t.data != nullptr && t.nbytes > 0 && pointer_on_device(t.data)) {
        need_stage = true;
        break;
      }
    }
    if (!need_stage && outputs != nullptr) {
      for (const auto& t : *outputs) {
        if (t.data != nullptr && t.nbytes > 0 && pointer_on_device(t.data)) {
          need_stage = true;
          break;
        }
      }
    }
  }
  if (need_stage) {
    return dispatch_via_host_staging(op_name, op_version, inputs, outputs, attrs);
  }
  return dispatch(op_name, op_version, inputs, outputs, attrs);
}

Error HostExecutableBackend::dispatch_via_host_staging(
    const std::string& op_name,
    const std::string& op_version,
    const std::vector<kernels::TensorView>& inputs,
    std::vector<kernels::TensorView>* outputs,
    const std::vector<ir::Attribute>& attrs) {
  if (outputs == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "dispatch outputs null");
  }

  std::vector<std::vector<std::uint8_t>> in_storage(inputs.size());
  std::vector<kernels::TensorView> host_inputs = inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].data == nullptr || inputs[i].nbytes == 0) {
      continue;
    }
    if (!pointer_on_device(inputs[i].data)) {
      // Already host-resident — run in place.
      continue;
    }
    in_storage[i].resize(inputs[i].nbytes);
    Error err = copy_d2h(inputs[i].data, in_storage[i].data(), inputs[i].nbytes);
    if (!err.ok()) {
      return err;
    }
    host_inputs[i].data = in_storage[i].data();
  }

  std::vector<std::vector<std::uint8_t>> out_storage(outputs->size());
  std::vector<bool> out_on_device(outputs->size(), false);
  std::vector<kernels::TensorView> host_outputs = *outputs;
  for (std::size_t i = 0; i < outputs->size(); ++i) {
    if ((*outputs)[i].data == nullptr || (*outputs)[i].nbytes == 0) {
      continue;
    }
    if (!pointer_on_device((*outputs)[i].data)) {
      continue;
    }
    out_on_device[i] = true;
    out_storage[i].resize((*outputs)[i].nbytes);
    // Seed with device contents when kernels may read-modify-write.
    Error err = copy_d2h((*outputs)[i].data, out_storage[i].data(), (*outputs)[i].nbytes);
    if (!err.ok()) {
      return err;
    }
    host_outputs[i].data = out_storage[i].data();
  }

  Error err =
      kernels::dispatch_cpu(op_name, op_version, host_inputs, &host_outputs, attrs);
  if (!err.ok()) {
    return err;
  }

  for (std::size_t i = 0; i < outputs->size(); ++i) {
    if (!out_on_device[i] || (*outputs)[i].data == nullptr || (*outputs)[i].nbytes == 0) {
      continue;
    }
    err = copy_h2d(out_storage[i].data(), (*outputs)[i].data, (*outputs)[i].nbytes);
    if (!err.ok()) {
      return err;
    }
  }
  return Error::success();
}

}  // namespace backends
}  // namespace uaii
