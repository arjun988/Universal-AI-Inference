#pragma once

#include "uaii/export.hpp"

#include <sstream>
#include <string>
#include <utility>

namespace uaii {
namespace log {

enum class Level {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
  Off = 5,
};

UAII_API void set_level(Level level) noexcept;
[[nodiscard]] UAII_API Level level() noexcept;

UAII_API void set_use_color(bool enabled) noexcept;

/// Write a log line. Thread-safe.
UAII_API void write(Level level, const char* component, const std::string& message);

[[nodiscard]] UAII_API const char* to_string(Level level) noexcept;

namespace detail {

class LogLine {
 public:
  LogLine(Level level, const char* component)
      : level_(level), component_(component) {}

  ~LogLine() {
    write(level_, component_, stream_.str());
  }

  template <typename T>
  LogLine& operator<<(const T& value) {
    stream_ << value;
    return *this;
  }

 private:
  Level level_;
  const char* component_;
  std::ostringstream stream_;
};

}  // namespace detail

inline detail::LogLine trace(const char* component) {
  return detail::LogLine{Level::Trace, component};
}
inline detail::LogLine debug(const char* component) {
  return detail::LogLine{Level::Debug, component};
}
inline detail::LogLine info(const char* component) {
  return detail::LogLine{Level::Info, component};
}
inline detail::LogLine warn(const char* component) {
  return detail::LogLine{Level::Warn, component};
}
inline detail::LogLine error(const char* component) {
  return detail::LogLine{Level::Error, component};
}

/// Parse level name: trace|debug|info|warn|error|off (case-insensitive).
[[nodiscard]] UAII_API bool parse_level(const std::string& text, Level* out) noexcept;

}  // namespace log
}  // namespace uaii
