#include "uaii/memory/budget.hpp"

namespace uaii {
namespace memory {

Error MemoryBudget::reserve(std::size_t bytes) {
  const std::uint64_t need = static_cast<std::uint64_t>(bytes);
  if (limit_bytes_ != 0 && used_bytes_ + need > limit_bytes_) {
    return Error::make(ErrorCode::InvalidArgument,
                       "memory budget exceeded: used=" + std::to_string(used_bytes_) +
                           " need=" + std::to_string(need) +
                           " limit=" + std::to_string(limit_bytes_));
  }
  used_bytes_ += need;
  if (used_bytes_ > peak_bytes_) {
    peak_bytes_ = used_bytes_;
  }
  return Error::ok();
}

void MemoryBudget::release(std::size_t bytes) noexcept {
  const std::uint64_t n = static_cast<std::uint64_t>(bytes);
  if (n >= used_bytes_) {
    used_bytes_ = 0;
  } else {
    used_bytes_ -= n;
  }
}

}  // namespace memory
}  // namespace uaii
