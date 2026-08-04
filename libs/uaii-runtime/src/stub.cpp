#include "uaii/core/error.hpp"

namespace uaii {
namespace runtime {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-runtime is a Phase 3 module (stub in Phase 1)");
}

}  // namespace runtime
}  // namespace uaii
