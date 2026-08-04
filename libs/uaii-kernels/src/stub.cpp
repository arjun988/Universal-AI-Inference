#include "uaii/core/error.hpp"

namespace uaii {
namespace kernels {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-kernels is a Phase 3 module (stub in Phase 1)");
}

}  // namespace kernels
}  // namespace uaii
