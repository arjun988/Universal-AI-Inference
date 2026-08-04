#include "uaii/version.hpp"

namespace uaii {

Version version() noexcept {
  return Version{};
}

const char* version_string() noexcept {
  return UAII_VERSION_STRING;
}

}  // namespace uaii
