#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/memory/arena.hpp"
#include "uaii/memory/budget.hpp"
#include "uaii/memory/buffer.hpp"
#include "uaii/memory/pool.hpp"

#include <string>

namespace uaii {
namespace memory {

enum class AllocStrategy {
  Pool = 0,
  Arena = 1,
};

struct AllocatorConfig {
  AllocStrategy strategy = AllocStrategy::Pool;
  std::uint64_t budget_bytes = 0;  // 0 = unlimited
  std::size_t arena_chunk_bytes = 1 << 20;
};

/// High-level allocator used by the runtime session.
class UAII_API Allocator {
 public:
  explicit Allocator(AllocatorConfig config = {});

  [[nodiscard]] MemoryBudget& budget() noexcept { return budget_; }
  [[nodiscard]] const MemoryBudget& budget() const noexcept { return budget_; }

  [[nodiscard]] Error allocate_bytes(std::size_t bytes, void** out);
  void deallocate_bytes(void* ptr, std::size_t bytes) noexcept;

  [[nodiscard]] Error allocate_tensor(const ir::Tensor& tensor, TensorBuffer* out);
  void deallocate_tensor(TensorBuffer* buffer) noexcept;

  void reset() noexcept;

  [[nodiscard]] std::string stats() const;

 private:
  AllocatorConfig config_;
  MemoryBudget budget_;
  TensorPool pool_;
  ArenaAllocator arena_;
};

}  // namespace memory
}  // namespace uaii
