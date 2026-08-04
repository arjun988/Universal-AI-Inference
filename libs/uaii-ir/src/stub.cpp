#include "uaii/core/error.hpp"

namespace uaii {
namespace ir {

/// Phase 2 will implement the UAII IR graph here.
Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-ir is a Phase 2 module (stub in Phase 1)");
}

}  // namespace ir
}  // namespace uaii
