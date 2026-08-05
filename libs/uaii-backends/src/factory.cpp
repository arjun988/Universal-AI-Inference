#include "uaii/backends/factory.hpp"

#include "uaii/backends/cpu_backend.hpp"
#include "uaii/backends/cuda_backend.hpp"
#include "uaii/backends/metal_backend.hpp"
#include "uaii/backends/rocm_backend.hpp"
#include "uaii/backends/vulkan_backend.hpp"
#include "uaii/backends/webgpu_backend.hpp"

#include <algorithm>
#include <cctype>

namespace uaii {
namespace backends {
namespace {

std::string normalize_name(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return name;
}

}  // namespace

std::vector<BackendDescriptor> list_backends() {
  return {
      {"cpu", DeviceType::Cpu, true, true,
       "Host CPU backend (reference kernels)"},
      {"cuda", DeviceType::Cuda, true, CudaBackend::native_compiled(),
       "CUDA backend (host-fallback always; native when UAII_WITH_CUDA=ON)"},
      {"metal", DeviceType::Metal, true, MetalBackend::native_compiled(),
       "Metal backend (host-fallback always; native when UAII_WITH_METAL=ON)"},
      {"vulkan", DeviceType::Vulkan, true, VulkanBackend::native_compiled(),
       "Vulkan backend (host-fallback always; native when UAII_WITH_VULKAN=ON)"},
      {"webgpu", DeviceType::WebGpu, true, WebGpuBackend::native_compiled(),
       "WebGPU backend (host-fallback always; native when UAII_WITH_WEBGPU=ON)"},
      {"rocm", DeviceType::Rocm, true, RocmBackend::native_compiled(),
       "ROCm backend (host-fallback always; native when UAII_WITH_ROCM=ON)"},
  };
}

bool backend_exists(const std::string& name) noexcept {
  const std::string n = normalize_name(name);
  for (const auto& d : list_backends()) {
    if (d.name == n) {
      return true;
    }
  }
  return false;
}

Error create_backend(const std::string& name,
                     const BackendCreateOptions& options,
                     std::unique_ptr<IBackend>* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "backend out null");
  }
  out->reset();

  const std::string n = normalize_name(name);
  const bool force = options.force_host_fallback || !options.prefer_native;

  if (n == "cpu") {
    *out = std::make_unique<CpuBackend>(options.allocator);
  } else if (n == "cuda") {
    *out = std::make_unique<CudaBackend>(options.allocator, force);
  } else if (n == "metal") {
    *out = std::make_unique<MetalBackend>(options.allocator, force);
  } else if (n == "vulkan") {
    *out = std::make_unique<VulkanBackend>(options.allocator, force);
  } else if (n == "webgpu") {
    *out = std::make_unique<WebGpuBackend>(options.allocator, force);
  } else if (n == "rocm") {
    *out = std::make_unique<RocmBackend>(options.allocator, force);
  } else {
    return Error::make(ErrorCode::NotFound, "unknown backend '" + name +
                                                "' (cpu|cuda|metal|vulkan|webgpu|rocm)");
  }
  return Error::success();
}

const char* device_type_backend_name(DeviceType type) noexcept {
  return to_string(type);
}

}  // namespace backends
}  // namespace uaii
