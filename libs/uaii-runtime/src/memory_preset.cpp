#include "uaii/runtime/memory_preset.hpp"

#include "uaii/core/log.hpp"
#include "uaii/kernels/thread_pool.hpp"

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
#  include <TargetConditionals.h>
#  include <mach/mach.h>
#  if TARGET_OS_IPHONE
#    include <os/proc.h>
#  endif
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
  if (n == "phone") {
    *out = MemoryPreset::Phone;
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
      out->kv_window = 2048;
      out->kv_sink = 4;
      out->enable_streaming = true;
      out->label = "laptop";
      break;
    case MemoryPreset::Desktop:
      out->ram_budget_bytes = gib(32);
      out->pin_layers = 16;
      out->expert_cache_bytes = gib(4);
      out->kv_window = 4096;
      out->kv_sink = 4;
      out->enable_streaming = true;
      out->label = "desktop";
      break;
    case MemoryPreset::Server:
      out->ram_budget_bytes = gib(128);
      out->pin_layers = 0;  // keep all dense when budget allows
      out->expert_cache_bytes = gib(32);
      out->kv_window = 8192;
      out->kv_sink = 4;
      out->enable_streaming = true;
      out->label = "server";
      break;
    case MemoryPreset::Phone:
      out->ram_budget_bytes = gib(2);
      out->pin_layers = 0;
      out->expert_cache_bytes = 256ull * 1024ull * 1024ull;
      out->kv_window = 512;
      out->kv_sink = 4;
      out->enable_streaming = true;
      out->single_staging = true;
      out->max_threads = 4;
      out->label = "phone";
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
  apply_available_memory_clamp(out, system_available_bytes());
  return Error::success();
}

std::uint64_t clamp_budget_to_available(std::uint64_t requested,
                                       std::uint64_t available) noexcept {
  if (available == 0 || requested == 0) return requested;
  const std::uint64_t half = available / 2ull;
  return requested < half ? requested : half;
}

void apply_available_memory_clamp(MemoryPresetConfig* cfg,
                                  std::uint64_t available) noexcept {
  if (cfg == nullptr) return;
  const std::uint64_t requested = cfg->ram_budget_bytes;
  if (available == 0 || requested == 0) {
    cfg->budget_clamped = false;
    return;
  }
  // Phone leaves half of free RAM for the OS and the rest of the app.
  // Laptop and above already name a resident target, so they shrink only when
  // that target is larger than the memory the OS can still give this process.
  const std::uint64_t cap = cfg->preset == MemoryPreset::Phone
                                ? clamp_budget_to_available(requested, available)
                                : (requested < available ? requested : available);
  if (requested <= cap) {
    cfg->budget_clamped = false;
    return;
  }
  cfg->budget_clamped = true;
  cfg->ram_budget_bytes = cap;
  constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;
  if (cap < kGiB) {
    if (cfg->kv_window > 128) cfg->kv_window = 128;
  } else if (cap < kGiB + kGiB / 2ull) {
    if (cfg->kv_window > 256) cfg->kv_window = 256;
  }
  const std::uint64_t expert_cap = cap / 8ull;
  if (cfg->expert_cache_bytes > expert_cap) cfg->expert_cache_bytes = expert_cap;
}

void apply_memory_preset(const MemoryPresetConfig& cfg, SessionOptions* opts) {
  if (!opts) return;
  if (cfg.ram_budget_bytes > 0) {
    opts->allocator.budget_bytes = cfg.ram_budget_bytes;
  }
  opts->enable_streaming = cfg.enable_streaming;
  opts->pin_layers = cfg.pin_layers;
  opts->expert_cache_bytes = cfg.expert_cache_bytes;
  opts->kv_window = cfg.kv_window;
  opts->kv_sink = cfg.kv_sink;
  opts->single_staging = cfg.single_staging;
  opts->max_threads = cfg.max_threads;
  kernels::set_thread_limit(cfg.max_threads > 0 ? static_cast<unsigned>(cfg.max_threads) : 0u);
  if (!cfg.label.empty()) {
    log::info("memory") << "preset=" << cfg.label
                        << " budget_bytes=" << cfg.ram_budget_bytes
                        << " pin_layers=" << cfg.pin_layers
                        << " kv_window=" << cfg.kv_window
                        << " staging_slots=" << (cfg.single_staging ? 1 : 2)
                        << " threads=" << cfg.max_threads
                        << " clamped=" << (cfg.budget_clamped ? "yes" : "no")
                        << " streaming=" << (cfg.enable_streaming ? "on" : "off");
  }
}

std::uint64_t system_available_bytes() noexcept {
#if defined(_WIN32)
  MEMORYSTATUSEX st{};
  st.dwLength = sizeof(st);
  if (!GlobalMemoryStatusEx(&st)) return 0;
  return static_cast<std::uint64_t>(st.ullAvailPhys);
#elif defined(__APPLE__)
#  if TARGET_OS_IPHONE
  return static_cast<std::uint64_t>(os_proc_available_memory());
#  else
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS) {
    return 0;
  }
  vm_size_t page = 0;
  if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS || page == 0) return 0;
  return static_cast<std::uint64_t>(vm.free_count + vm.inactive_count) *
         static_cast<std::uint64_t>(page);
#  endif
#else
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  unsigned long long kb = 0;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::sscanf(line, "MemAvailable: %llu", &kb) == 1) break;
  }
  std::fclose(f);
  return static_cast<std::uint64_t>(kb) * 1024ull;
#endif
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
