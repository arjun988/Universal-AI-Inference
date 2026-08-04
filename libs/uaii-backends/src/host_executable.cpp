#include "uaii/backends/host_executable.hpp"

#include "uaii/kernels/kernels.hpp"

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
  return Error::ok();
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
  caps.details = details_;
  return caps;
}

Error HostExecutableBackend::allocate(std::size_t bytes, void** out_ptr) {
  Error err = allocator_->allocate_bytes(bytes, out_ptr);
  if (!err.ok()) {
    return err;
  }
  orphan_allocs_.emplace_back(*out_ptr, bytes);
  return Error::ok();
}

Error HostExecutableBackend::free(void* ptr) noexcept {
  if (ptr == nullptr) {
    return Error::ok();
  }
  for (auto it = orphan_allocs_.begin(); it != orphan_allocs_.end(); ++it) {
    if (it->first == ptr) {
      allocator_->deallocate_bytes(it->first, it->second);
      orphan_allocs_.erase(it);
      return Error::ok();
    }
  }
  return Error::make(ErrorCode::NotFound, "pointer not owned by backend");
}

Error HostExecutableBackend::synchronize() {
  return Error::ok();
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

}  // namespace backends
}  // namespace uaii
