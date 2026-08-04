#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/backend.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/kernels/tensor_view.hpp"
#include "uaii/memory/allocator.hpp"

#include <memory>
#include <string>
#include <vector>

namespace uaii {
namespace backends {

/// CPU device backend: host memory + kernel dispatch.
class UAII_API CpuBackend : public IBackend {
 public:
  explicit CpuBackend(memory::Allocator* allocator = nullptr);
  ~CpuBackend() override;

  [[nodiscard]] std::string name() const override { return "cpu"; }
  [[nodiscard]] DeviceType device_type() const override { return DeviceType::Cpu; }

  [[nodiscard]] Error initialize() override;
  void shutdown() noexcept override;

  [[nodiscard]] BackendCapabilities capabilities() const override;

  [[nodiscard]] Error allocate(std::size_t bytes, void** out_ptr) override;
  [[nodiscard]] Error free(void* ptr) noexcept override;

  [[nodiscard]] Error synchronize() override;

  [[nodiscard]] Error dispatch(const std::string& op_name,
                               const std::string& op_version,
                               const std::vector<kernels::TensorView>& inputs,
                               std::vector<kernels::TensorView>* outputs,
                               const std::vector<ir::Attribute>& attrs);

  [[nodiscard]] memory::Allocator* allocator() noexcept { return allocator_; }

 private:
  memory::Allocator* allocator_ = nullptr;
  std::unique_ptr<memory::Allocator> owned_allocator_;
  bool initialized_ = false;
  // Track orphan allocations when using owned allocator via IBackend::allocate.
  std::vector<std::pair<void*, std::size_t>> orphan_allocs_;
};

}  // namespace backends
}  // namespace uaii
