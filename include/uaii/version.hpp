#pragma once

#include "uaii/version_generated.hpp"
#include "uaii/export.hpp"

namespace uaii {

struct Version {
  int major = UAII_VERSION_MAJOR;
  int minor = UAII_VERSION_MINOR;
  int patch = UAII_VERSION_PATCH;
};

/// Compile-time / runtime library version.
[[nodiscard]] UAII_API Version version() noexcept;

/// Human-readable version string, e.g. "0.1.0".
[[nodiscard]] UAII_API const char* version_string() noexcept;

}  // namespace uaii
