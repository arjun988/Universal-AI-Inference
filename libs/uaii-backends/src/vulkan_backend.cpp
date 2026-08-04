#include "uaii/backends/vulkan_backend.hpp"

#include "native_stubs.hpp"

namespace uaii {
namespace backends {

VulkanBackend::VulkanBackend(memory::Allocator* allocator, bool force_host_fallback)
    : HostExecutableBackend("vulkan", DeviceType::Vulkan, allocator,
                            /*host_fallback=*/true),
      force_host_fallback_(force_host_fallback) {
  set_details("Vulkan backend (host-fallback)");
  set_native_available(false);
}

bool VulkanBackend::native_compiled() noexcept {
  return native::vulkan_compiled();
}

Error VulkanBackend::initialize() {
  Error err = HostExecutableBackend::initialize();
  if (!err.ok()) {
    return err;
  }

  native_ready_ = false;
  set_native_available(false);
  set_host_fallback(true);

  if (!force_host_fallback_ && native::vulkan_compiled()) {
    err = native::vulkan_init();
    if (err.ok()) {
      native_ready_ = true;
      set_native_available(true);
      set_host_fallback(false);
      set_details("Vulkan backend (native scaffold / device path)");
      return Error::ok();
    }
    set_details("Vulkan backend (host-fallback; native init failed: " + err.message() +
                ")");
  } else if (force_host_fallback_) {
    set_details("Vulkan backend (forced host-fallback)");
  } else {
    set_details(
        "Vulkan backend (host-fallback; build with -DUAII_WITH_VULKAN=ON for native)");
  }
  return Error::ok();
}

Error VulkanBackend::dispatch(const std::string& op_name,
                              const std::string& op_version,
                              const std::vector<kernels::TensorView>& inputs,
                              std::vector<kernels::TensorView>* outputs,
                              const std::vector<ir::Attribute>& attrs) {
  if (native_ready_) {
    Error err = native::vulkan_dispatch(op_name, op_version, inputs, outputs, attrs);
    if (err.ok()) {
      return err;
    }
  }
  return HostExecutableBackend::dispatch(op_name, op_version, inputs, outputs, attrs);
}

}  // namespace backends
}  // namespace uaii
