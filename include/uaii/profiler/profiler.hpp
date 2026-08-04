#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace uaii {
namespace profiler {

enum class EventCategory {
  Kernel = 0,
  Io,
  Memory,
  Planner,
  Other,
};

struct ProfileEvent {
  std::string name;
  EventCategory category = EventCategory::Other;
  std::int64_t ts_us = 0;       // start, microseconds from session epoch
  std::int64_t dur_us = 0;      // duration
  std::string detail;
};

class UAII_API Profiler {
 public:
  void begin_session(std::string name = "uaii");
  void end_session();

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] const std::string& session_name() const noexcept { return name_; }
  [[nodiscard]] const std::vector<ProfileEvent>& events() const noexcept { return events_; }

  void clear() noexcept;

  /// Record a completed interval.
  void add_event(std::string name,
                 EventCategory category,
                 std::int64_t start_us,
                 std::int64_t dur_us,
                 std::string detail = {});

  [[nodiscard]] std::int64_t now_us() const;

  /// RAII scope that records on destruction when profiler is active.
  class Scope {
   public:
    Scope(Profiler* p, std::string name, EventCategory cat, std::string detail = {});
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    Profiler* p_ = nullptr;
    std::string name_;
    EventCategory cat_ = EventCategory::Other;
    std::string detail_;
    std::int64_t start_us_ = 0;
  };

  [[nodiscard]] std::string summary() const;

 private:
  bool active_ = false;
  std::string name_;
  using Clock = std::chrono::steady_clock;
  Clock::time_point epoch_{};
  std::vector<ProfileEvent> events_;
  mutable std::mutex mu_;
};

[[nodiscard]] UAII_API const char* to_string(EventCategory c) noexcept;

/// Write Chrome Trace Event Format JSON (view in chrome://tracing or Perfetto).
[[nodiscard]] UAII_API Error write_chrome_trace(const Profiler& profiler,
                                                const std::string& path);

}  // namespace profiler
}  // namespace uaii
