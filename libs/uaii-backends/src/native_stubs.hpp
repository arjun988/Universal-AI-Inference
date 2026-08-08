#pragma once

// Optional native hooks. When UAII_WITH_* is ON, corresponding .cpp/.cu files
// provide real implementations; otherwise native_unavailable.cpp stubs return errors.

#include "uaii/core/error.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace uaii {
namespace backends {
namespace native {

[[nodiscard]] bool cuda_compiled() noexcept;
[[nodiscard]] Error cuda_init();
void cuda_shutdown() noexcept;
[[nodiscard]] Error cuda_allocate(std::size_t bytes, void** out_ptr);
[[nodiscard]] Error cuda_free(void* ptr) noexcept;
[[nodiscard]] Error cuda_copy_h2d(const void* host, void* device, std::size_t bytes);
[[nodiscard]] Error cuda_copy_h2d_async(const void* host, void* device, std::size_t bytes);
[[nodiscard]] bool cuda_probe_device() noexcept;
/// True when `p` is CUDA device (or managed) memory.
[[nodiscard]] bool cuda_is_device_ptr(const void* p) noexcept;
[[nodiscard]] Error cuda_copy_d2h(const void* device, void* host, std::size_t bytes);
[[nodiscard]] Error cuda_copy_d2d(const void* src, void* dst, std::size_t bytes);
[[nodiscard]] Error cuda_synchronize();
[[nodiscard]] Error cuda_dispatch(const std::string& op_name,
                                  const std::string& op_version,
                                  const std::vector<kernels::TensorView>& inputs,
                                  std::vector<kernels::TensorView>* outputs,
                                  const std::vector<ir::Attribute>& attrs);
/// Human-readable capability note (ops on-device vs host_fallback).
[[nodiscard]] const char* cuda_capability_details() noexcept;

[[nodiscard]] bool metal_compiled() noexcept;
[[nodiscard]] Error metal_init();
void metal_shutdown() noexcept;
[[nodiscard]] Error metal_allocate(std::size_t bytes, void** out_ptr);
[[nodiscard]] Error metal_free(void* ptr) noexcept;
[[nodiscard]] Error metal_copy_h2d(const void* host, void* device, std::size_t bytes);
[[nodiscard]] Error metal_copy_d2h(const void* device, void* host, std::size_t bytes);
[[nodiscard]] Error metal_copy_d2d(const void* src, void* dst, std::size_t bytes);
[[nodiscard]] Error metal_synchronize();
[[nodiscard]] Error metal_dispatch(const std::string& op_name,
                                   const std::string& op_version,
                                   const std::vector<kernels::TensorView>& inputs,
                                   std::vector<kernels::TensorView>* outputs,
                                   const std::vector<ir::Attribute>& attrs);
[[nodiscard]] const char* metal_capability_details() noexcept;

[[nodiscard]] bool vulkan_compiled() noexcept;
[[nodiscard]] Error vulkan_init();
void vulkan_shutdown() noexcept;
[[nodiscard]] Error vulkan_allocate(std::size_t bytes, void** out_ptr);
[[nodiscard]] Error vulkan_free(void* ptr) noexcept;
[[nodiscard]] Error vulkan_copy_h2d(const void* host, void* device, std::size_t bytes);
[[nodiscard]] Error vulkan_copy_d2h(const void* device, void* host, std::size_t bytes);
[[nodiscard]] Error vulkan_copy_d2d(const void* src, void* dst, std::size_t bytes);
[[nodiscard]] Error vulkan_synchronize();
[[nodiscard]] Error vulkan_dispatch(const std::string& op_name,
                                    const std::string& op_version,
                                    const std::vector<kernels::TensorView>& inputs,
                                    std::vector<kernels::TensorView>* outputs,
                                    const std::vector<ir::Attribute>& attrs);
[[nodiscard]] const char* vulkan_capability_details() noexcept;

[[nodiscard]] bool webgpu_compiled() noexcept;
[[nodiscard]] Error webgpu_init();
void webgpu_shutdown() noexcept;
[[nodiscard]] Error webgpu_allocate(std::size_t bytes, void** out_ptr);
[[nodiscard]] Error webgpu_free(void* ptr) noexcept;
[[nodiscard]] Error webgpu_copy_h2d(const void* host, void* device, std::size_t bytes);
[[nodiscard]] Error webgpu_copy_d2h(const void* device, void* host, std::size_t bytes);
[[nodiscard]] Error webgpu_copy_d2d(const void* src, void* dst, std::size_t bytes);
[[nodiscard]] Error webgpu_synchronize();
[[nodiscard]] Error webgpu_dispatch(const std::string& op_name,
                                    const std::string& op_version,
                                    const std::vector<kernels::TensorView>& inputs,
                                    std::vector<kernels::TensorView>* outputs,
                                    const std::vector<ir::Attribute>& attrs);
[[nodiscard]] const char* webgpu_capability_details() noexcept;

[[nodiscard]] bool rocm_compiled() noexcept;
[[nodiscard]] Error rocm_init();
void rocm_shutdown() noexcept;
[[nodiscard]] Error rocm_allocate(std::size_t bytes, void** out_ptr);
[[nodiscard]] Error rocm_free(void* ptr) noexcept;
[[nodiscard]] Error rocm_copy_h2d(const void* host, void* device, std::size_t bytes);
[[nodiscard]] Error rocm_copy_d2h(const void* device, void* host, std::size_t bytes);
[[nodiscard]] Error rocm_copy_d2d(const void* src, void* dst, std::size_t bytes);
[[nodiscard]] Error rocm_synchronize();
[[nodiscard]] Error rocm_dispatch(const std::string& op_name,
                                  const std::string& op_version,
                                  const std::vector<kernels::TensorView>& inputs,
                                  std::vector<kernels::TensorView>* outputs,
                                  const std::vector<ir::Attribute>& attrs);
[[nodiscard]] const char* rocm_capability_details() noexcept;

}  // namespace native
}  // namespace backends
}  // namespace uaii
