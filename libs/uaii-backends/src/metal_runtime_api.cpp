// Non-Apple / no-ObjC Metal path: clear errors. Apple builds use metal_runtime_api.mm.

#include "native_stubs.hpp"

namespace uaii {
namespace backends {
namespace native {

bool metal_compiled() noexcept { return true; }

const char* metal_capability_details() noexcept {
  return "Metal: device path unavailable on this platform (host_fallback: all ops)";
}

Error metal_init() {
  return Error::make(ErrorCode::NotImplemented,
                     "Metal native path requires Apple + Objective-C++ (metal_runtime_api.mm)");
}

void metal_shutdown() noexcept {}

Error metal_allocate(std::size_t, void**) {
  return metal_init();
}

Error metal_free(void*) noexcept { return Error::success(); }

Error metal_copy_h2d(const void*, void*, std::size_t) { return metal_init(); }
Error metal_copy_d2h(const void*, void*, std::size_t) { return metal_init(); }
Error metal_copy_d2d(const void*, void*, std::size_t) { return metal_init(); }
Error metal_synchronize() { return metal_init(); }

Error metal_dispatch(const std::string&, const std::string&,
                     const std::vector<kernels::TensorView>&,
                     std::vector<kernels::TensorView>*,
                     const std::vector<ir::Attribute>&) {
  return Error::make(ErrorCode::NotImplemented, "Metal native dispatch unavailable");
}

}  // namespace native
}  // namespace backends
}  // namespace uaii
