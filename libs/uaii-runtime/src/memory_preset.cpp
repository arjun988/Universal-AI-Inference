#include "uaii/runtime/memory_preset.hpp"

#include "uaii/core/log.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#else
#  include <cstdio>
#endif

namespace uaii {
namespace runtime {
namespace {

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

}  // namespace

bool parse_memory_preset(const std::string& name, MemoryPreset* out) {
  if (!out) return false;
  const std::string n = lower(name);
  if (n == "none" || n.empty()) {
    *out = MemoryPreset::None;
    return true;
  }
  if (n == "laptop") {
    *out = MemoryPreset::Laptop;
    return true;
  }
  if (n == "desktop") {
    *out = MemoryPreset::Desktop;
    return true;
  }
  if (n == "server") {
    *out = MemoryPreset::Server;
    return true;
  }
  return false;
}

Error resolve_memory_preset(MemoryPreset preset, std::uint64_t ram_gb_override,
                            MemoryPresetConfig* out) {
  if (!out) return Error::make(ErrorCode::InvalidArgument, "MemoryPresetConfig out null");
  *out = MemoryPresetConfig{};
  out->preset = preset;

  auto gib = [](std::uint64_t n) -> std::uint64_t { return n * 1024ull * 1024ull * 1024ull; };

  switch (preset) {
    case MemoryPreset::Laptop:
      out->ram_budget_bytes = gib(8);
      out->pin_layers = 2;
      out->expert_cache_bytes = gib(1);
      out->enable_streaming = true;
      out->label = "laptop";
      break;
    case MemoryPreset::Desktop:
      out->ram_budget_bytes = gib(32);
      out->pin_layers = 16;
      out->expert_cache_bytes = gib(4);
      out->enable_streaming = true;
      out->label = "desktop";
      break;
    case MemoryPreset::Server:
      out->ram_budget_bytes = gib(128);
      out->pin_layers = 0;  // keep all dense when budget allows
      out->expert_cache_bytes = gib(32);
      out->enable_streaming = true;
      out->label = "server";
      break;
    case MemoryPreset::None:
    default:
      out->enable_streaming = false;
      out->label = "none";
      break;
  }

  if (ram_gb_override > 0) {
    out->ram_budget_bytes = gib(ram_gb_override);
    out->enable_streaming = true;
    if (out->label == "none") out->label = "custom";
  }
  return Error::success();
}

void apply_memory_preset(const MemoryPresetConfig& cfg, SessionOptions* opts) {
  if (!opts) return;
  if (cfg.ram_budget_bytes > 0) {
    opts->allocator.budget_bytes = cfg.ram_budget_bytes;
  }
  opts->enable_streaming = cfg.enable_streaming;
  opts->pin_layers = cfg.pin_layers;
  opts->expert_cache_bytes = cfg.expert_cache_bytes;
  if (!cfg.label.empty()) {
    log::info("memory") << "preset=" << cfg.label
                        << " budget_bytes=" << cfg.ram_budget_bytes
                        << " pin_layers=" << cfg.pin_layers
                        << " streaming=" << (cfg.enable_streaming ? "on" : "off");
  }
}

std::uint64_t process_current_rss_bytes() noexcept {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  pmc.cb = sizeof(pmc);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
  return static_cast<std::uint64_t>(pmc.WorkingSetSize);
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::uint64_t>(info.resident_size);
#else
  FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return 0;
  char line[256];
  std::uint64_t kb = 0;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::sscanf(line, "VmRSS: %llu", reinterpret_cast<unsigned long long*>(&kb)) == 1) {
      break;
    }
  }
  std::fclose(f);
  return kb * 1024ull;
#endif
}

std::uint64_t process_peak_rss_bytes() noexcept {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  pmc.cb = sizeof(pmc);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
  return static_cast<std::uint64_t>(pmc.PeakWorkingSetSize);
#elif defined(__APPLE__)
  return process_current_rss_bytes();
#else
  FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return 0;
  char line[256];
  std::uint64_t kb = 0;
  while (std::fgets(line, sizeof(line), f)) {
    unsigned long long v = 0;
    if (std::sscanf(line, "VmHWM: %llu", &v) == 1) {
      kb = static_cast<std::uint64_t>(v);
      break;
    }
  }
  std::fclose(f);
  return kb * 1024ull;
#endif
}

}  // namespace runtime
}  // namespace uaii
