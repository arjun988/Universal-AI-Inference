#include "uaii/backends/cpu_backend.hpp"

#include "uaii/kernels/kernels.hpp"

namespace uaii {
namespace backends {

CpuBackend::CpuBackend(memory::Allocator* allocator) : allocator_(allocator) {
  if (allocator_ == nullptr) {
    owned_allocator_ = std::make_unique<memory::Allocator>();
    allocator_ = owned_allocator_.get();
  }
}

CpuBackend::~CpuBackend() {
  shutdown();
}

Error CpuBackend::initialize() {
  initialized_ = true;
  return Error::ok();
}

void CpuBackend::shutdown() noexcept {
  for (auto& kv : orphan_allocs_) {
    allocator_->deallocate_bytes(kv.first, kv.second);
  }
  orphan_allocs_.clear();
  initialized_ = false;
}

BackendCapabilities CpuBackend::capabilities() const {
  BackendCapabilities caps;
  caps.name = "cpu";
  caps.device_type = DeviceType::Cpu;
  caps.supported_dtypes = {DType::F32};
  caps.supports_profiling = false;
  caps.supports_async = false;
  caps.details = "Phase 3 host CPU backend (f32 kernels)";
  return caps;
}

Error CpuBackend::allocate(std::size_t bytes, void** out_ptr) {
  Error err = allocator_->allocate_bytes(bytes, out_ptr);
  if (!err.ok()) {
    return err;
  }
  orphan_allocs_.emplace_back(*out_ptr, bytes);
  return Error::ok();
}

Error CpuBackend::free(void* ptr) noexcept {
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
  return Error::make(ErrorCode::NotFound, "pointer not owned by CpuBackend");
}

Error CpuBackend::synchronize() {
  // CPU kernels are synchronous.
  return Error::ok();
}

Error CpuBackend::dispatch(const std::string& op_name,
                           const std::string& op_version,
                           const std::vector<kernels::TensorView>& inputs,
                           std::vector<kernels::TensorView>* outputs,
                           const std::vector<ir::Attribute>& attrs) {
  if (!initialized_) {
    return Error::make(ErrorCode::InvalidArgument, "CpuBackend not initialized");
  }
  return kernels::dispatch_cpu(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
