#include "native_stubs.hpp"

// Default implementations when vendor SDKs are not enabled at build time.

#ifndef UAII_WITH_CUDA
namespace uaii {
namespace backends {
namespace native {
bool cuda_compiled() noexcept { return false; }
Error cuda_init() {
  return Error::make(ErrorCode::NotImplemented,
                     "CUDA native path not compiled (configure -DUAII_WITH_CUDA=ON)");
}
void cuda_shutdown() noexcept {}
Error cuda_dispatch(const std::string&, const std::string&,
                    const std::vector<kernels::TensorView>&,
                    std::vector<kernels::TensorView>*,
                    const std::vector<ir::Attribute>&) {
  return cuda_init();
}
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_METAL
namespace uaii {
namespace backends {
namespace native {
bool metal_compiled() noexcept { return false; }
Error metal_init() {
  return Error::make(ErrorCode::NotImplemented,
                     "Metal native path not compiled (configure -DUAII_WITH_METAL=ON)");
}
void metal_shutdown() noexcept {}
Error metal_dispatch(const std::string&, const std::string&,
                     const std::vector<kernels::TensorView>&,
                     std::vector<kernels::TensorView>*,
                     const std::vector<ir::Attribute>&) {
  return metal_init();
}
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_VULKAN
namespace uaii {
namespace backends {
namespace native {
bool vulkan_compiled() noexcept { return false; }
Error vulkan_init() {
  return Error::make(ErrorCode::NotImplemented,
                     "Vulkan native path not compiled (configure -DUAII_WITH_VULKAN=ON)");
}
void vulkan_shutdown() noexcept {}
Error vulkan_dispatch(const std::string&, const std::string&,
                      const std::vector<kernels::TensorView>&,
                      std::vector<kernels::TensorView>*,
                      const std::vector<ir::Attribute>&) {
  return vulkan_init();
}
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_WEBGPU
namespace uaii {
namespace backends {
namespace native {
bool webgpu_compiled() noexcept { return false; }
Error webgpu_init() {
  return Error::make(ErrorCode::NotImplemented,
                     "WebGPU native path not compiled (configure -DUAII_WITH_WEBGPU=ON)");
}
void webgpu_shutdown() noexcept {}
Error webgpu_dispatch(const std::string&, const std::string&,
                      const std::vector<kernels::TensorView>&,
                      std::vector<kernels::TensorView>*,
                      const std::vector<ir::Attribute>&) {
  return webgpu_init();
}
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_ROCM
namespace uaii {
namespace backends {
namespace native {
bool rocm_compiled() noexcept { return false; }
Error rocm_init() {
  return Error::make(ErrorCode::NotImplemented,
                     "ROCm native path not compiled (configure -DUAII_WITH_ROCM=ON)");
}
void rocm_shutdown() noexcept {}
Error rocm_dispatch(const std::string&, const std::string&,
                    const std::vector<kernels::TensorView>&,
                    std::vector<kernels::TensorView>*,
                    const std::vector<ir::Attribute>&) {
  return rocm_init();
}
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif
