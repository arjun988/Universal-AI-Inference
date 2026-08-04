#include "uaii/c_api/plugin_abi.h"

#include <cstddef>
#include <cstring>

namespace {

uaii_plugin_info g_info{};

void copy_cstr(char* dst, std::size_t dst_size, const char* src) {
  if (dst_size == 0) {
    return;
  }
  std::strncpy(dst, src, dst_size - 1);
  dst[dst_size - 1] = '\0';
}

void fill_info() {
  std::memset(&g_info, 0, sizeof(g_info));
  g_info.abi_version = UAII_PLUGIN_ABI_VERSION;
  g_info.kind = UAII_PLUGIN_KIND_PROBE;
  copy_cstr(g_info.name, UAII_PLUGIN_NAME_MAX, "example_probe");
  copy_cstr(g_info.version, UAII_PLUGIN_VERSION_MAX, "0.1.0");
  copy_cstr(g_info.description, UAII_PLUGIN_DESC_MAX,
            "Phase 1 example plugin used to validate discovery and loading");
}

}  // namespace

extern "C" {

#if defined(_WIN32)
__declspec(dllexport)
#endif
const uaii_plugin_info* uaii_plugin_get_info(void) {
  if (g_info.abi_version == 0) {
    fill_info();
  }
  return &g_info;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
int uaii_plugin_init(void) {
  if (g_info.abi_version == 0) {
    fill_info();
  }
  return 0;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
void uaii_plugin_shutdown(void) {
  // No resources to release in the probe plugin.
}

}  // extern "C"
