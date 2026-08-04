#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/memory/budget.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace uaii {
namespace memory {

/// Bump-pointer arena. Fast allocate; free only via reset().
class UAII_API ArenaAllocator {
 public:
  explicit ArenaAllocator(MemoryBudget* budget = nullptr,
                          std::size_t chunk_bytes = 1 << 20);

  ArenaAllocator(const ArenaAllocator&) = delete;
  ArenaAllocator& operator=(const ArenaAllocator&) = delete;

  [[nodiscard]] Error allocate(std::size_t bytes, std::size_t alignment, void** out);
  void reset() noexcept;

  [[nodiscard]] std::size_t bytes_allocated() const noexcept { return allocated_; }
  [[nodiscard]] std::size_t chunk_count() const noexcept { return chunks_.size(); }

 private:
  struct Chunk {
    std::unique_ptr<std::uint8_t[]> data;
    std::size_t capacity = 0;
    std::size_t offset = 0;
  };

  [[nodiscard]] Error new_chunk(std::size_t min_bytes);

  MemoryBudget* budget_ = nullptr;
  std::size_t chunk_bytes_ = 1 << 20;
  std::vector<Chunk> chunks_;
  std::size_t allocated_ = 0;
};

}  // namespace memory
}  // namespace uaii
