#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/memory/budget.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace memory {

/// Size-classed tensor buffer pool with reuse.
class UAII_API TensorPool {
 public:
  explicit TensorPool(MemoryBudget* budget = nullptr);

  TensorPool(const TensorPool&) = delete;
  TensorPool& operator=(const TensorPool&) = delete;

  [[nodiscard]] Error allocate(std::size_t bytes, void** out);
  void deallocate(void* ptr, std::size_t bytes) noexcept;
  void clear() noexcept;

  [[nodiscard]] std::size_t live_bytes() const noexcept { return live_bytes_; }
  [[nodiscard]] std::size_t cached_bytes() const noexcept { return cached_bytes_; }

 private:
  static std::size_t round_up_size(std::size_t bytes) noexcept;

  MemoryBudget* budget_ = nullptr;
  std::mutex mutex_;
  std::unordered_map<std::size_t, std::vector<void*>> free_lists_;
  std::unordered_map<void*, std::size_t> live_;
  std::size_t live_bytes_ = 0;
  std::size_t cached_bytes_ = 0;
};

}  // namespace memory
}  // namespace uaii
