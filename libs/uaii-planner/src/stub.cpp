#include "uaii/core/error.hpp"

namespace uaii {
namespace planner {

Error not_implemented_yet() {
  return Error::make(ErrorCode::NotImplemented,
                     "uaii-planner is a Phase 3/6 module (stub in Phase 1)");
}

}  // namespace planner
}  // namespace uaii
