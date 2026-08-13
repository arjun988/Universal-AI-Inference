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
};

struct MemoryPresetConfig {
  MemoryPreset preset = MemoryPreset::None;
  std::uint64_t ram_budget_bytes = 0;
  /// Pin first N transformer layers resident; stream the rest (0 = auto from budget).
  std::int64_t pin_layers = 0;
  /// Expert LRU capacity in bytes (MoE); 0 = auto.
  std::uint64_t expert_cache_bytes = 0;
  bool enable_streaming = false;
  std::string label;
};

[[nodiscard]] UAII_API bool parse_memory_preset(const std::string& name, MemoryPreset* out);

[[nodiscard]] UAII_API Error resolve_memory_preset(MemoryPreset preset,
                                                   std::uint64_t ram_gb_override,
                                                   MemoryPresetConfig* out);

/// Apply preset fields onto SessionOptions (budget + streaming + pin_layers metadata).
UAII_API void apply_memory_preset(const MemoryPresetConfig& cfg, SessionOptions* opts);

/// Current process peak / current RSS in bytes (0 if unavailable).
[[nodiscard]] UAII_API std::uint64_t process_current_rss_bytes() noexcept;
[[nodiscard]] UAII_API std::uint64_t process_peak_rss_bytes() noexcept;

}  // namespace runtime
}  // namespace uaii
