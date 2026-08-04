#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <cstddef>
#include <cstdint>

namespace uaii {
namespace memory {

/// Tracks allocation against an optional hard budget.
class UAII_API MemoryBudget {
 public:
  explicit MemoryBudget(std::uint64_t limit_bytes = 0) : limit_bytes_(limit_bytes) {}

  void set_limit(std::uint64_t limit_bytes) noexcept { limit_bytes_ = limit_bytes; }
  [[nodiscard]] std::uint64_t limit() const noexcept { return limit_bytes_; }
  [[nodiscard]] std::uint64_t used() const noexcept { return used_bytes_; }
  [[nodiscard]] std::uint64_t peak() const noexcept { return peak_bytes_; }

  [[nodiscard]] Error reserve(std::size_t bytes);
  void release(std::size_t bytes) noexcept;
  void reset_peak() noexcept { peak_bytes_ = used_bytes_; }

 private:
  std::uint64_t limit_bytes_ = 0;  // 0 = unlimited
  std::uint64_t used_bytes_ = 0;
  std::uint64_t peak_bytes_ = 0;
};

}  // namespace memory
}  // namespace uaii
