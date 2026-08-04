#include "uaii/core/error.hpp"

namespace uaii {
namespace memory {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-memory is a Phase 3 module (stub in Phase 1)");
}

}  // namespace memory
}  // namespace uaii
