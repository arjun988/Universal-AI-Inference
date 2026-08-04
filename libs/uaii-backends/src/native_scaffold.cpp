#include "native_stubs.hpp"

#include "uaii/kernels/kernels.hpp"

// When UAII_WITH_* is ON, provide a native scaffold that proves backend wiring.
// Real device kernels can replace dispatch later without changing the IBackend API.
// No vendor SDK install is required for this scaffold.

#if defined(UAII_WITH_CUDA) || defined(UAII_WITH_METAL) || defined(UAII_WITH_VULKAN) || \
    defined(UAII_WITH_WEBGPU) || defined(UAII_WITH_ROCM)

namespace uaii {
namespace backends {
namespace native {
namespace {

Error scaffold_dispatch(const std::string& op_name,
                        const std::string& op_version,
                        const std::vector<kernels::TensorView>& inputs,
                        std::vector<kernels::TensorView>* outputs,
                        const std::vector<ir::Attribute>& attrs) {
  return kernels::dispatch_cpu(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace

#ifdef UAII_WITH_CUDA
bool cuda_compiled() noexcept { return true; }
Error cuda_init() { return Error::ok(); }
void cuda_shutdown() noexcept {}
Error cuda_dispatch(const std::string& op_name, const std::string& op_version,
                    const std::vector<kernels::TensorView>& inputs,
                    std::vector<kernels::TensorView>* outputs,
                    const std::vector<ir::Attribute>& attrs) {
  return scaffold_dispatch(op_name, op_version, inputs, outputs, attrs);
}
#endif

#ifdef UAII_WITH_METAL
bool metal_compiled() noexcept { return true; }
Error metal_init() { return Error::ok(); }
void metal_shutdown() noexcept {}
Error metal_dispatch(const std::string& op_name, const std::string& op_version,
                     const std::vector<kernels::TensorView>& inputs,
                     std::vector<kernels::TensorView>* outputs,
                     const std::vector<ir::Attribute>& attrs) {
  return scaffold_dispatch(op_name, op_version, inputs, outputs, attrs);
}
#endif

#ifdef UAII_WITH_VULKAN
bool vulkan_compiled() noexcept { return true; }
Error vulkan_init() { return Error::ok(); }
void vulkan_shutdown() noexcept {}
Error vulkan_dispatch(const std::string& op_name, const std::string& op_version,
                      const std::vector<kernels::TensorView>& inputs,
                      std::vector<kernels::TensorView>* outputs,
                      const std::vector<ir::Attribute>& attrs) {
  return scaffold_dispatch(op_name, op_version, inputs, outputs, attrs);
}
#endif

#ifdef UAII_WITH_WEBGPU
bool webgpu_compiled() noexcept { return true; }
Error webgpu_init() { return Error::ok(); }
void webgpu_shutdown() noexcept {}
Error webgpu_dispatch(const std::string& op_name, const std::string& op_version,
                      const std::vector<kernels::TensorView>& inputs,
                      std::vector<kernels::TensorView>* outputs,
                      const std::vector<ir::Attribute>& attrs) {
  return scaffold_dispatch(op_name, op_version, inputs, outputs, attrs);
}
#endif

#ifdef UAII_WITH_ROCM
bool rocm_compiled() noexcept { return true; }
Error rocm_init() { return Error::ok(); }
void rocm_shutdown() noexcept {}
Error rocm_dispatch(const std::string& op_name, const std::string& op_version,
                    const std::vector<kernels::TensorView>& inputs,
                    std::vector<kernels::TensorView>* outputs,
                    const std::vector<ir::Attribute>& attrs) {
  return scaffold_dispatch(op_name, op_version, inputs, outputs, attrs);
}
#endif

}  // namespace native
}  // namespace backends
}  // namespace uaii

#endif  // any UAII_WITH_*
