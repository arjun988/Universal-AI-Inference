#pragma once

// Optional native hooks. When UAII_WITH_* is ON, corresponding .cpp files
// provide real implementations; otherwise these weak stubs return unavailable.

#include "uaii/core/error.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <string>
#include <vector>

namespace uaii {
namespace backends {
namespace native {

[[nodiscard]] bool cuda_compiled() noexcept;
[[nodiscard]] Error cuda_init();
void cuda_shutdown() noexcept;
[[nodiscard]] Error cuda_dispatch(const std::string& op_name,
                                  const std::string& op_version,
                                  const std::vector<kernels::TensorView>& inputs,
                                  std::vector<kernels::TensorView>* outputs,
                                  const std::vector<ir::Attribute>& attrs);

[[nodiscard]] bool metal_compiled() noexcept;
[[nodiscard]] Error metal_init();
void metal_shutdown() noexcept;
[[nodiscard]] Error metal_dispatch(const std::string& op_name,
                                   const std::string& op_version,
                                   const std::vector<kernels::TensorView>& inputs,
                                   std::vector<kernels::TensorView>* outputs,
                                   const std::vector<ir::Attribute>& attrs);

[[nodiscard]] bool vulkan_compiled() noexcept;
[[nodiscard]] Error vulkan_init();
void vulkan_shutdown() noexcept;
[[nodiscard]] Error vulkan_dispatch(const std::string& op_name,
                                    const std::string& op_version,
                                    const std::vector<kernels::TensorView>& inputs,
                                    std::vector<kernels::TensorView>* outputs,
                                    const std::vector<ir::Attribute>& attrs);

[[nodiscard]] bool webgpu_compiled() noexcept;
[[nodiscard]] Error webgpu_init();
void webgpu_shutdown() noexcept;
[[nodiscard]] Error webgpu_dispatch(const std::string& op_name,
                                    const std::string& op_version,
                                    const std::vector<kernels::TensorView>& inputs,
                                    std::vector<kernels::TensorView>* outputs,
                                    const std::vector<ir::Attribute>& attrs);

[[nodiscard]] bool rocm_compiled() noexcept;
[[nodiscard]] Error rocm_init();
void rocm_shutdown() noexcept;
[[nodiscard]] Error rocm_dispatch(const std::string& op_name,
                                  const std::string& op_version,
                                  const std::vector<kernels::TensorView>& inputs,
                                  std::vector<kernels::TensorView>* outputs,
                                  const std::vector<ir::Attribute>& attrs);

}  // namespace native
}  // namespace backends
}  // namespace uaii
