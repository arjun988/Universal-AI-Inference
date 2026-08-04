#include "uaii/core/error.hpp"

namespace uaii {
namespace profiler {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-profiler is a Phase 6 module (stub in Phase 1)");
}

}  // namespace profiler
}  // namespace uaii
