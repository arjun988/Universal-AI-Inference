#include "uaii/memory/allocator.hpp"

#include "uaii/ir/dtype.hpp"

#include <sstream>

namespace uaii {
namespace memory {

Allocator::Allocator(AllocatorConfig config)
    : config_(config),
      budget_(config.budget_bytes),
      pool_(&budget_),
      arena_(&budget_, config.arena_chunk_bytes) {}

Error Allocator::allocate_bytes(std::size_t bytes, void** out) {
  if (config_.strategy == AllocStrategy::Arena) {
    return arena_.allocate(bytes, alignof(std::max_align_t), out);
  }
  return pool_.allocate(bytes, out);
}

void Allocator::deallocate_bytes(void* ptr, std::size_t bytes) noexcept {
  if (config_.strategy == AllocStrategy::Arena) {
    // Arena frees only on reset.
    (void)ptr;
    (void)bytes;
    return;
  }
  pool_.deallocate(ptr, bytes);
}

Error Allocator::allocate_tensor(const ir::Tensor& tensor, TensorBuffer* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tensor buffer out is null");
  }
  const std::uint64_t nbytes64 = ir::estimate_tensor_bytes(tensor);
  if (nbytes64 == 0) {
    return Error::make(ErrorCode::InvalidArgument,
                       "cannot allocate tensor with unknown size: " + tensor.name);
  }
  const std::size_t nbytes = static_cast<std::size_t>(nbytes64);
  void* ptr = nullptr;
  Error err = allocate_bytes(nbytes, &ptr);
  if (!err.ok()) {
    return err;
  }
  out->id = tensor.id;
  out->dtype = tensor.dtype;
  out->shape = tensor.shape;
  out->data = ptr;
  out->nbytes = nbytes;
  out->owned = true;
  return Error::success();
}

void Allocator::deallocate_tensor(TensorBuffer* buffer) noexcept {
  if (buffer == nullptr || !buffer->owned || buffer->data == nullptr) {
    return;
  }
  deallocate_bytes(buffer->data, buffer->nbytes);
  buffer->data = nullptr;
  buffer->nbytes = 0;
  buffer->owned = false;
}

void Allocator::reset() noexcept {
  if (config_.strategy == AllocStrategy::Arena) {
    arena_.reset();
  } else {
    pool_.clear();
  }
}

std::string Allocator::stats() const {
  std::ostringstream oss;
  oss << "budget_used=" << budget_.used() << " peak=" << budget_.peak();
  if (budget_.limit() != 0) {
    oss << " limit=" << budget_.limit();
  }
  if (config_.strategy == AllocStrategy::Pool) {
    oss << " pool_live=" << pool_.live_bytes() << " pool_cached=" << pool_.cached_bytes();
  } else {
    oss << " arena_alloc=" << arena_.bytes_allocated()
        << " arena_chunks=" << arena_.chunk_count();
  }
  return oss.str();
}

}  // namespace memory
}  // namespace uaii
