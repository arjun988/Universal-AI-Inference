#pragma once

#include "uaii/export.hpp"

#include <ostream>
#include <string>
#include <utility>

namespace uaii {

/// Stable error categories for CLI exit codes and SDK mapping.
enum class ErrorCode {
  Ok = 0,
  InvalidArgument = 1,
  NotFound = 2,
  IoError = 3,
  ConfigError = 4,
  PluginError = 5,
  AbiMismatch = 6,
  NotImplemented = 7,
  Internal = 8,
};

[[nodiscard]] UAII_API const char* to_string(ErrorCode code) noexcept;

/// Result-like error object. Empty message means success when code == Ok.
class UAII_API Error {
 public:
  Error() = default;
  Error(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  /// Factory for a success result (named distinctly from member `ok()` for GCC).
  [[nodiscard]] static Error success() { return Error{}; }

  [[nodiscard]] static Error make(ErrorCode code, std::string message) {
    return Error{code, std::move(message)};
  }

  [[nodiscard]] bool ok() const noexcept { return code_ == ErrorCode::Ok; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

  [[nodiscard]] std::string to_string() const;

 private:
  ErrorCode code_ = ErrorCode::Ok;
  std::string message_;
};

inline std::ostream& operator<<(std::ostream& os, const Error& err) {
  return os << err.to_string();
}

}  // namespace uaii
