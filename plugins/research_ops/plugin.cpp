#include "uaii/c_api/plugin_abi.h"
#include "uaii/plugins/operator_host.hpp"

#include <cstring>

namespace {

uaii_plugin_info g_info{};

constexpr const char* kScaffoldMsg =
    "scaffold — implement kernel or use supported decoder-only models";

void fill_info() {
  std::memset(&g_info, 0, sizeof(g_info));
  g_info.abi_version = UAII_PLUGIN_ABI_VERSION;
  g_info.kind = UAII_PLUGIN_KIND_OPERATOR;
  std::strncpy(g_info.name, "research_ops", UAII_PLUGIN_NAME_MAX - 1);
  std::strncpy(g_info.version, "0.1.0", UAII_PLUGIN_VERSION_MAX - 1);
  std::strncpy(g_info.description,
               "Fail-closed scaffolds for SSM/RWKV/vision/encoder ops (NotImplemented)",
               UAII_PLUGIN_DESC_MAX - 1);
}

uaii::Error scaffold_op(const std::vector<uaii::kernels::TensorView>&,
                        std::vector<uaii::kernels::TensorView>*,
                        const std::vector<uaii::ir::Attribute>&) {
  return uaii::Error::make(uaii::ErrorCode::NotImplemented, kScaffoldMsg);
}

}  // namespace

extern "C" {

#if defined(_WIN32)
__declspec(dllexport)
#endif
const uaii_plugin_info* uaii_plugin_get_info(void) {
  if (g_info.abi_version == 0) fill_info();
  return &g_info;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
int uaii_plugin_init(void) {
  if (g_info.abi_version == 0) fill_info();
  auto& reg = uaii::plugins::OperatorHostRegistry::instance();
  reg.register_op("SSMConv1d", scaffold_op);
  reg.register_op("SSMState", scaffold_op);
  reg.register_op("RWKVTimeMix", scaffold_op);
  reg.register_op("EncoderCrossAttn", scaffold_op);
  reg.register_op("VisionPatchEmbed", scaffold_op);
  return 0;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
void uaii_plugin_shutdown(void) {}

}  // extern "C"
