#include "uaii/c_api/plugin_abi.h"
#include "uaii/plugins/operator_host.hpp"

#include <cstring>

namespace {

uaii_plugin_info g_info{};

void fill_info() {
  std::memset(&g_info, 0, sizeof(g_info));
  g_info.abi_version = UAII_PLUGIN_ABI_VERSION;
  g_info.kind = UAII_PLUGIN_KIND_OPERATOR;
  std::strncpy(g_info.name, "example_op", UAII_PLUGIN_NAME_MAX - 1);
  std::strncpy(g_info.version, "0.1.0", UAII_PLUGIN_VERSION_MAX - 1);
  std::strncpy(g_info.description, "Registers Neg operator into CPU dispatch hot path",
               UAII_PLUGIN_DESC_MAX - 1);
}

uaii::Error neg_op(const std::vector<uaii::kernels::TensorView>& inputs,
                   std::vector<uaii::kernels::TensorView>* outputs,
                   const std::vector<uaii::ir::Attribute>&) {
  if (inputs.size() != 1 || outputs == nullptr || outputs->size() != 1) {
    return uaii::Error::make(uaii::ErrorCode::InvalidArgument, "Neg arity");
  }
  const auto& in = inputs[0];
  auto& out = (*outputs)[0];
  if (in.dtype != uaii::DType::F32 || out.dtype != uaii::DType::F32 ||
      in.data == nullptr || out.data == nullptr || in.numel() != out.numel()) {
    return uaii::Error::make(uaii::ErrorCode::InvalidArgument, "Neg f32 shape");
  }
  const float* x = in.f32();
  float* y = out.f32();
  for (std::size_t i = 0; i < in.numel(); ++i) y[i] = -x[i];
  return uaii::Error::success();
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
  uaii::plugins::OperatorHostRegistry::instance().register_op("Neg", neg_op);
  return 0;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
void uaii_plugin_shutdown(void) {}

}  // extern "C"
