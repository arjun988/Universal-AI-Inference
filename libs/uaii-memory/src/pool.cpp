#include "uaii/memory/pool.hpp"

#include <cstdlib>

namespace uaii {
namespace memory {

TensorPool::TensorPool(MemoryBudget* budget) : budget_(budget) {}

std::size_t TensorPool::round_up_size(std::size_t bytes) noexcept {
  // Power-of-two size classes from 64B upward.
  std::size_t n = 64;
  while (n < bytes) {
    if (n > (std::size_t{1} << 30)) {
      return bytes;
    }
    n <<= 1;
  }
  return n;
}

Error TensorPool::allocate(std::size_t bytes, void** out) {
  if (out == nullptr || bytes == 0) {
    return Error::make(ErrorCode::InvalidArgument, "pool allocate invalid args");
  }
  const std::size_t rounded = round_up_size(bytes);
  std::lock_guard<std::mutex> lock(mutex_);

  auto& list = free_lists_[rounded];
  void* ptr = nullptr;
  if (!list.empty()) {
    ptr = list.back();
    list.pop_back();
    cached_bytes_ -= rounded;
  } else {
    if (budget_ != nullptr) {
      Error err = budget_->reserve(rounded);
      if (!err.ok()) {
        return err;
      }
    }
    ptr = std::malloc(rounded);
    if (ptr == nullptr) {
      if (budget_ != nullptr) {
        budget_->release(rounded);
      }
      return Error::make(ErrorCode::IoError, "malloc failed");
    }
  }

  live_[ptr] = rounded;
  live_bytes_ += rounded;
  *out = ptr;
  return Error::ok();
}

void TensorPool::deallocate(void* ptr, std::size_t /*bytes*/) noexcept {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = live_.find(ptr);
  if (it == live_.end()) {
    return;
  }
  const std::size_t rounded = it->second;
  live_.erase(it);
  live_bytes_ -= rounded;
  free_lists_[rounded].push_back(ptr);
  cached_bytes_ += rounded;
}

void TensorPool::clear() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& kv : live_) {
    std::free(kv.first);
    if (budget_ != nullptr) {
      budget_->release(kv.second);
    }
  }
  live_.clear();
  live_bytes_ = 0;

  for (auto& kv : free_lists_) {
    for (void* ptr : kv.second) {
      std::free(ptr);
      if (budget_ != nullptr) {
        budget_->release(kv.first);
      }
    }
  }
  free_lists_.clear();
  cached_bytes_ = 0;
}

}  // namespace memory
}  // namespace uaii
