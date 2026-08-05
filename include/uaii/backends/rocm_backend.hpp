#pragma once

#include "uaii/backends/host_executable.hpp"
#include "uaii/export.hpp"

#include <utility>
#include <vector>

namespace uaii {
namespace backends {

class UAII_API RocmBackend : public HostExecutableBackend {
 public:
  explicit RocmBackend(memory::Allocator* allocator = nullptr,
                       bool force_host_fallback = false);
  ~RocmBackend() override;

  [[nodiscard]] Error initialize() override;
  void shutdown() noexcept override;

  [[nodiscard]] Error allocate(std::size_t bytes, void** out_ptr) override;
  [[nodiscard]] Error free(void* ptr) noexcept override;
  [[nodiscard]] Error copy_h2d(const void* host, void* device, std::size_t bytes) override;
  [[nodiscard]] Error copy_d2h(const void* device, void* host, std::size_t bytes) override;
  [[nodiscard]] Error copy_d2d(const void* src, void* dst, std::size_t bytes) override;
  [[nodiscard]] Error synchronize() override;

  [[nodiscard]] Error dispatch(const std::string& op_name,
                               const std::string& op_version,
                               const std::vector<kernels::TensorView>& inputs,
                               std::vector<kernels::TensorView>* outputs,
                               const std::vector<ir::Attribute>& attrs) override;

  [[nodiscard]] static bool native_compiled() noexcept;

 private:
  bool force_host_fallback_ = false;
  bool native_ready_ = false;
  std::vector<std::pair<void*, std::size_t>> device_allocs_;
};

}  // namespace backends
}  // namespace uaii
