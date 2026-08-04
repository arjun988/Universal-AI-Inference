#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/backend.hpp"
#include "uaii/memory/allocator.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace uaii {
namespace backends {

/// Shared host-memory + CPU-kernel execution path.
/// Used by CpuBackend and by GPU backends in host-fallback mode (no SDK required).
class UAII_API HostExecutableBackend : public IBackend {
 public:
  HostExecutableBackend(std::string name,
                        DeviceType device_type,
                        memory::Allocator* allocator = nullptr,
                        bool host_fallback = false);

  ~HostExecutableBackend() override;

  [[nodiscard]] std::string name() const override { return name_; }
  [[nodiscard]] DeviceType device_type() const override { return device_type_; }
  [[nodiscard]] bool uses_host_fallback() const noexcept override { return host_fallback_; }

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
                               const std::vector<ir::Attribute>& attrs) override;

  [[nodiscard]] memory::Allocator* allocator() noexcept { return allocator_; }

  void set_details(std::string details) { details_ = std::move(details); }
  void set_native_available(bool v) noexcept { native_available_ = v; }
  void set_host_fallback(bool v) noexcept { host_fallback_ = v; }

 protected:
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }

 private:
  std::string name_;
  DeviceType device_type_ = DeviceType::Unknown;
  bool host_fallback_ = false;
  bool native_available_ = false;
  std::string details_;
  memory::Allocator* allocator_ = nullptr;
  std::unique_ptr<memory::Allocator> owned_allocator_;
  bool initialized_ = false;
  std::vector<std::pair<void*, std::size_t>> orphan_allocs_;
};

}  // namespace backends
}  // namespace uaii
