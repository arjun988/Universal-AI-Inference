#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/runtime/session.hpp"

#include <cstdint>
#include <string>

namespace uaii {
namespace runtime {

enum class MemoryPreset {
  None = 0,
  Laptop,   // ~8 GiB resident target
  Desktop,  // ~32 GiB
  Server,   // ~128 GiB
  Phone,    // ~2 GiB process cap (Android / iOS)
};

struct MemoryPresetConfig {
  MemoryPreset preset = MemoryPreset::None;
  std::uint64_t ram_budget_bytes = 0;
  /// Pin first N transformer layers resident; stream the rest (0 = auto from budget).
  std::int64_t pin_layers = 0;
  /// Expert LRU capacity in bytes (MoE); 0 = auto.
  std::uint64_t expert_cache_bytes = 0;
  /// Sliding KV window (tokens). 0 = unset / do not apply.
  std::int64_t kv_window = 0;
  std::int64_t kv_sink = 4;
  bool enable_streaming = false;
  /// Phone keeps one staging buffer. Laptop and above prefetch into a second.
  bool single_staging = false;
  /// 0 = use hardware_concurrency (still honors UAII_NUM_THREADS). Phone uses 4.
  int max_threads = 0;
  /// True when the OS available-RAM cap reduced ram_budget_bytes.
  bool budget_clamped = false;
  std::string label;
};

[[nodiscard]] UAII_API bool parse_memory_preset(const std::string& name, MemoryPreset* out);

[[nodiscard]] UAII_API Error resolve_memory_preset(MemoryPreset preset,
                                                   std::uint64_t ram_gb_override,
                                                   MemoryPresetConfig* out);

/// Apply preset fields onto SessionOptions (budget + streaming + pin_layers metadata).
UAII_API void apply_memory_preset(const MemoryPresetConfig& cfg, SessionOptions* opts);

/// Bytes the OS will still give this process (0 if unknown).
/// Phones use the jetsam / MemAvailable figure, not the spec-sheet RAM.
[[nodiscard]] UAII_API std::uint64_t system_available_bytes() noexcept;

/// `requested` is kept when `available` is 0. Otherwise the cap is half of `available`.
[[nodiscard]] UAII_API std::uint64_t clamp_budget_to_available(std::uint64_t requested,
                                                              std::uint64_t available) noexcept;

/// Shrink budget, KV window, and expert cache so a typed --ram-gb cannot exceed the device.
UAII_API void apply_available_memory_clamp(MemoryPresetConfig* cfg,
                                           std::uint64_t available) noexcept;

/// Current process peak / current RSS in bytes (0 if unavailable).
[[nodiscard]] UAII_API std::uint64_t process_current_rss_bytes() noexcept;
[[nodiscard]] UAII_API std::uint64_t process_peak_rss_bytes() noexcept;

}  // namespace runtime
}  // namespace uaii
