#include "uaii/profiler/profiler.hpp"

#include <fstream>
#include <sstream>

namespace uaii {
namespace profiler {

const char* to_string(EventCategory c) noexcept {
  switch (c) {
    case EventCategory::Kernel: return "kernel";
    case EventCategory::Io: return "io";
    case EventCategory::Memory: return "memory";
    case EventCategory::Planner: return "planner";
    default: return "other";
  }
}

void Profiler::begin_session(std::string name) {
  std::lock_guard<std::mutex> lock(mu_);
  name_ = std::move(name);
  events_.clear();
  epoch_ = Clock::now();
  active_ = true;
}

void Profiler::end_session() {
  std::lock_guard<std::mutex> lock(mu_);
  active_ = false;
}

void Profiler::clear() noexcept {
  std::lock_guard<std::mutex> lock(mu_);
  events_.clear();
  active_ = false;
}

std::int64_t Profiler::now_us() const {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - epoch_)
      .count();
}

void Profiler::add_event(std::string name,
                         EventCategory category,
                         std::int64_t start_us,
                         std::int64_t dur_us,
                         std::string detail) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!active_) return;
  ProfileEvent e;
  e.name = std::move(name);
  e.category = category;
  e.ts_us = start_us;
  e.dur_us = dur_us;
  e.detail = std::move(detail);
  events_.push_back(std::move(e));
}

Profiler::Scope::Scope(Profiler* p, std::string name, EventCategory cat, std::string detail)
    : p_(p), name_(std::move(name)), cat_(cat), detail_(std::move(detail)) {
  if (p_ && p_->active()) {
    start_us_ = p_->now_us();
  }
}

Profiler::Scope::~Scope() {
  if (p_ && p_->active()) {
    const std::int64_t end = p_->now_us();
    p_->add_event(std::move(name_), cat_, start_us_, end - start_us_, std::move(detail_));
  }
}

std::string Profiler::summary() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::int64_t kernel = 0, io = 0, mem = 0, plan = 0, other = 0;
  for (const auto& e : events_) {
    switch (e.category) {
      case EventCategory::Kernel: kernel += e.dur_us; break;
      case EventCategory::Io: io += e.dur_us; break;
      case EventCategory::Memory: mem += e.dur_us; break;
      case EventCategory::Planner: plan += e.dur_us; break;
      default: other += e.dur_us; break;
    }
  }
  std::ostringstream oss;
  oss << "events=" << events_.size() << " kernel_us=" << kernel << " io_us=" << io
      << " memory_us=" << mem << " planner_us=" << plan << " other_us=" << other;
  return oss.str();
}

Error write_chrome_trace(const Profiler& profiler, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Error::make(ErrorCode::IoError, "failed to write " + path);
  }
  out << "{\"traceEvents\":[\n";
  const auto& events = profiler.events();
  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto& e = events[i];
    if (i) out << ",\n";
    // Escape minimal JSON
    auto esc = [](const std::string& s) {
      std::string r;
      r.reserve(s.size());
      for (char c : s) {
        if (c == '"' || c == '\\') r.push_back('\\');
        r.push_back(c);
      }
      return r;
    };
    out << "  {\"name\":\"" << esc(e.name) << "\",\"cat\":\"" << to_string(e.category)
        << "\",\"ph\":\"X\",\"ts\":" << e.ts_us << ",\"dur\":" << e.dur_us
        << ",\"pid\":1,\"tid\":1,\"args\":{\"detail\":\"" << esc(e.detail) << "\"}}";
  }
  out << "\n],\"displayTimeUnit\":\"us\"}\n";
  return Error::ok();
}

}  // namespace profiler
}  // namespace uaii
