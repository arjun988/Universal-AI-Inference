#pragma once

#include "uaii/backends/host_executable.hpp"
#include "uaii/export.hpp"

#include <utility>
#include <vector>

namespace uaii {
namespace backends {

/// CUDA backend. Native device path (cuBLAS + kernels) when UAII_WITH_CUDA=ON;
/// otherwise host-fallback. Attention remains host_fallback on the native path.
class UAII_API CudaBackend : public HostExecutableBackend {
 public:
  explicit CudaBackend(memory::Allocator* allocator = nullptr,
                       bool force_host_fallback = false);

  ~CudaBackend() override;

  [[nodiscard]] Error initialize() override;
  void shutdown() noexcept override;

  [[nodiscard]] Error allocate(std::size_t bytes, void** out_ptr) override;
  [[nodiscard]] Error free(void* ptr) noexcept override;
  [[nodiscard]] BackendCapabilities capabilities() const override;

  [[nodiscard]] Error copy_h2d(const void* host, void* device, std::size_t bytes) override;
  [[nodiscard]] Error copy_h2d_async(const void* host, void* device,
                                     std::size_t bytes) override;
  [[nodiscard]] Error copy_d2h(const void* device, void* host, std::size_t bytes) override;
  [[nodiscard]] Error copy_d2d(const void* src, void* dst, std::size_t bytes) override;
  [[nodiscard]] Error synchronize() override;

  [[nodiscard]] Error dispatch(const std::string& op_name,
                               const std::string& op_version,
                               const std::vector<kernels::TensorView>& inputs,
                               std::vector<kernels::TensorView>* outputs,
                               const std::vector<ir::Attribute>& attrs) override;

  [[nodiscard]] static bool native_compiled() noexcept;
  /// True when built with CUDA and at least one device is present.
  [[nodiscard]] static bool native_device_available() noexcept;

 private:
  bool force_host_fallback_ = false;
  bool native_ready_ = false;
  std::vector<std::pair<void*, std::size_t>> device_allocs_;
};

}  // namespace backends
}  // namespace uaii
