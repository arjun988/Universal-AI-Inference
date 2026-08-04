#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <cstdint>
#include <string>

namespace uaii {
namespace ir {

/// UAII IR versioning (independent of library semver).
struct IrVersion {
  std::uint32_t major = 1;
  std::uint32_t minor = 0;

  [[nodiscard]] constexpr bool operator==(const IrVersion& o) const noexcept {
    return major == o.major && minor == o.minor;
  }
  [[nodiscard]] constexpr bool operator!=(const IrVersion& o) const noexcept {
    return !(*this == o);
  }

  /// Lexicographic compare for compatibility checks.
  [[nodiscard]] constexpr bool operator<(const IrVersion& o) const noexcept {
    return major < o.major || (major == o.major && minor < o.minor);
  }
};

/// Current IR produced by this library.
inline constexpr IrVersion kCurrentIrVersion{1, 0};

/// Oldest IR major.minor this library can load (same major, minor >=).
inline constexpr IrVersion kMinSupportedIrVersion{1, 0};

[[nodiscard]] inline std::string to_string(const IrVersion& v) {
  return std::to_string(v.major) + "." + std::to_string(v.minor);
}

/// Compatible when major matches and minor <= current minor.
[[nodiscard]] UAII_API bool is_compatible(const IrVersion& file_version) noexcept;

[[nodiscard]] UAII_API Error check_compatible(const IrVersion& file_version);

}  // namespace ir
}  // namespace uaii
