#include "uaii/memory/arena.hpp"

#include <algorithm>

namespace uaii {
namespace memory {
namespace {

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

ArenaAllocator::ArenaAllocator(MemoryBudget* budget, std::size_t chunk_bytes)
    : budget_(budget), chunk_bytes_(std::max<std::size_t>(chunk_bytes, 4096)) {}

Error ArenaAllocator::new_chunk(std::size_t min_bytes) {
  const std::size_t cap = std::max(chunk_bytes_, min_bytes);
  if (budget_ != nullptr) {
    Error err = budget_->reserve(cap);
    if (!err.ok()) {
      return err;
    }
  }
  Chunk chunk;
  chunk.data = std::make_unique<std::uint8_t[]>(cap);
  chunk.capacity = cap;
  chunk.offset = 0;
  chunks_.push_back(std::move(chunk));
  return Error::ok();
}

Error ArenaAllocator::allocate(std::size_t bytes, std::size_t alignment, void** out) {
  if (out == nullptr || bytes == 0) {
    return Error::make(ErrorCode::InvalidArgument, "arena allocate invalid args");
  }
  if (alignment == 0) {
    alignment = alignof(std::max_align_t);
  }

  if (chunks_.empty()) {
    Error err = new_chunk(bytes + alignment);
    if (!err.ok()) {
      return err;
    }
  }

  for (int pass = 0; pass < 2; ++pass) {
    Chunk& chunk = chunks_.back();
    const std::size_t aligned = align_up(chunk.offset, alignment);
    if (aligned + bytes <= chunk.capacity) {
      void* ptr = chunk.data.get() + aligned;
      chunk.offset = aligned + bytes;
      allocated_ += bytes;
      *out = ptr;
      return Error::ok();
    }
    if (pass == 0) {
      Error err = new_chunk(bytes + alignment);
      if (!err.ok()) {
        return err;
      }
    }
  }
  return Error::make(ErrorCode::Internal, "arena allocation failed");
}

void ArenaAllocator::reset() noexcept {
  for (auto& chunk : chunks_) {
    if (budget_ != nullptr) {
      budget_->release(chunk.capacity);
    }
  }
  chunks_.clear();
  allocated_ = 0;
}

}  // namespace memory
}  // namespace uaii
