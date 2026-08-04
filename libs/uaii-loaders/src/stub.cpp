#include "uaii/core/error.hpp"

namespace uaii {
namespace loaders {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-loaders is a Phase 4 module (stub in Phase 1)");
}

}  // namespace loaders
}  // namespace uaii
