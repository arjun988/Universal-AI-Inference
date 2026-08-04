#include "uaii/core/log.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace uaii {
namespace log {
namespace {

std::atomic<int> g_level{static_cast<int>(Level::Info)};
std::atomic<bool> g_color{true};
std::mutex g_mutex;

const char* ansi_for(Level level) {
  switch (level) {
    case Level::Trace: return "\033[90m";
    case Level::Debug: return "\033[36m";
    case Level::Info: return "\033[32m";
    case Level::Warn: return "\033[33m";
    case Level::Error: return "\033[31m";
    default: return "\033[0m";
  }
}

std::string timestamp_now() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const std::time_t t = clock::to_time_t(now);
  std::tm tm_buf{};
#if defined(_WIN32)
  localtime_s(&tm_buf, &t);
#else
  localtime_r(&t, &tm_buf);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

}  // namespace

void set_level(Level level) noexcept {
  g_level.store(static_cast<int>(level), std::memory_order_relaxed);
}

Level level() noexcept {
  return static_cast<Level>(g_level.load(std::memory_order_relaxed));
}

void set_use_color(bool enabled) noexcept {
  g_color.store(enabled, std::memory_order_relaxed);
}

const char* to_string(Level lvl) noexcept {
  switch (lvl) {
    case Level::Trace: return "trace";
    case Level::Debug: return "debug";
    case Level::Info: return "info";
    case Level::Warn: return "warn";
    case Level::Error: return "error";
    case Level::Off: return "off";
  }
  return "unknown";
}

bool parse_level(const std::string& text, Level* out) noexcept {
  if (out == nullptr) {
    return false;
  }
  std::string lower;
  lower.reserve(text.size());
  for (char c : text) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower == "trace") { *out = Level::Trace; return true; }
  if (lower == "debug") { *out = Level::Debug; return true; }
  if (lower == "info") { *out = Level::Info; return true; }
  if (lower == "warn" || lower == "warning") { *out = Level::Warn; return true; }
  if (lower == "error") { *out = Level::Error; return true; }
  if (lower == "off") { *out = Level::Off; return true; }
  return false;
}

void write(Level lvl, const char* component, const std::string& message) {
  if (static_cast<int>(lvl) < g_level.load(std::memory_order_relaxed)) {
    return;
  }
  if (lvl == Level::Off) {
    return;
  }

  const char* comp = component != nullptr ? component : "uaii";
  std::lock_guard<std::mutex> lock(g_mutex);

  const bool color = g_color.load(std::memory_order_relaxed);
  std::ostream& os = (lvl == Level::Error || lvl == Level::Warn) ? std::cerr : std::cout;

  if (color) {
    os << ansi_for(lvl);
  }
  os << timestamp_now() << " [" << to_string(lvl) << "] [" << comp << "] " << message;
  if (color) {
    os << "\033[0m";
  }
  os << '\n';
}

}  // namespace log
}  // namespace uaii
