#include "uaii/core/error.hpp"

namespace uaii {
namespace storage {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-storage is a Phase 3/6 module (stub in Phase 1)");
}

}  // namespace storage
}  // namespace uaii
