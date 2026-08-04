#include "uaii/core/error.hpp"

namespace uaii {
namespace backends {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-backends is a Phase 3/5 module (stub in Phase 1)");
}

}  // namespace backends
}  // namespace uaii
