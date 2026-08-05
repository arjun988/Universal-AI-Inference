#include "uaii/c_api/plugin_host.h"
#include "uaii/core/plugin.hpp"
#include "uaii/plugins/operator_host.hpp"

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace uaii {
namespace plugins {
namespace {

void* lookup(void* handle, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

int host_register_op(const char* op_name, uaii_host_op_fn fn, void* userdata) {
  if (op_name == nullptr || fn == nullptr) return 1;
  OperatorHostRegistry::instance().register_op(
      op_name,
      [fn, userdata](const std::vector<kernels::TensorView>& inputs,
                     std::vector<kernels::TensorView>* outputs,
                     const std::vector<ir::Attribute>& attrs) -> Error {
        const int rc = fn(static_cast<const void*>(&inputs), static_cast<void*>(outputs),
                          static_cast<const void*>(&attrs), userdata);
        if (rc != 0) {
          return Error::make(ErrorCode::PluginError,
                             std::string("plugin op '") + "failed code " +
                                 std::to_string(rc));
        }
        return Error::success();
      });
  return 0;
}

Error offer_host(void* library_handle, const PluginDescriptor& desc) {
  if (library_handle == nullptr) {
    return Error::make(ErrorCode::PluginError, "null plugin handle");
  }
  auto register_fn = reinterpret_cast<uaii_plugin_register_fn>(
      lookup(library_handle, "uaii_plugin_register"));
  if (register_fn == nullptr) {
    // C++ plugins (e.g. example_op) register inside uaii_plugin_init.
    (void)desc;
    return Error::success();
  }
  uaii_host_api host{};
  host.struct_size = static_cast<uint32_t>(sizeof(uaii_host_api));
  host.abi_version = UAII_PLUGIN_ABI_VERSION;
  host.register_op = &host_register_op;
  host.register_loader_probe = nullptr;
  const int rc = register_fn(&host);
  if (rc != 0) {
    return Error::make(ErrorCode::PluginError,
                       "uaii_plugin_register failed for " + desc.name + " code " +
                           std::to_string(rc));
  }
  return Error::success();
}

struct HostOfferInstaller {
  HostOfferInstaller() { set_plugin_host_offer(&offer_host); }
};

// Ensure the offer hook is installed whenever kernels are linked (CLI/runtime).
HostOfferInstaller g_install;

}  // namespace
}  // namespace plugins
}  // namespace uaii
