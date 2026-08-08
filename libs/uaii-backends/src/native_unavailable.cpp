#include "native_stubs.hpp"

// Default implementations when vendor SDKs are not enabled at build time.

#define UAII_NATIVE_UNAVAIL(prefix, Label)                                            \
  bool prefix##_compiled() noexcept { return false; }                                 \
  const char* prefix##_capability_details() noexcept {                                \
    return Label " host-fallback (native not compiled)";                              \
  }                                                                                   \
  Error prefix##_init() {                                                             \
    return Error::make(ErrorCode::NotImplemented,                                     \
                       Label " native path not compiled (configure -DUAII_WITH_" Label \
                             "=ON)");                                                 \
  }                                                                                   \
  void prefix##_shutdown() noexcept {}                                                \
  Error prefix##_allocate(std::size_t, void**) { return prefix##_init(); }            \
  Error prefix##_free(void*) noexcept { return Error::success(); }                    \
  Error prefix##_copy_h2d(const void*, void*, std::size_t) { return prefix##_init(); } \
  Error prefix##_copy_d2h(const void*, void*, std::size_t) { return prefix##_init(); } \
  Error prefix##_copy_d2d(const void*, void*, std::size_t) { return prefix##_init(); } \
  Error prefix##_synchronize() { return prefix##_init(); }                            \
  Error prefix##_dispatch(const std::string&, const std::string&,                     \
                          const std::vector<kernels::TensorView>&,                    \
                          std::vector<kernels::TensorView>*,                          \
                          const std::vector<ir::Attribute>&) {                        \
    return prefix##_init();                                                           \
  }

#ifndef UAII_WITH_CUDA
namespace uaii {
namespace backends {
namespace native {
UAII_NATIVE_UNAVAIL(cuda, "CUDA")
bool cuda_probe_device() noexcept { return false; }
bool cuda_is_device_ptr(const void*) noexcept { return false; }
Error cuda_copy_h2d_async(const void*, void*, std::size_t) { return cuda_init(); }
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_METAL
namespace uaii {
namespace backends {
namespace native {
UAII_NATIVE_UNAVAIL(metal, "METAL")
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_VULKAN
namespace uaii {
namespace backends {
namespace native {
UAII_NATIVE_UNAVAIL(vulkan, "VULKAN")
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_WEBGPU
namespace uaii {
namespace backends {
namespace native {
UAII_NATIVE_UNAVAIL(webgpu, "WEBGPU")
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#ifndef UAII_WITH_ROCM
namespace uaii {
namespace backends {
namespace native {
UAII_NATIVE_UNAVAIL(rocm, "ROCM")
}  // namespace native
}  // namespace backends
}  // namespace uaii
#endif

#undef UAII_NATIVE_UNAVAIL
