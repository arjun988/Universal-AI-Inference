#pragma once

#include "uaii/interfaces/types.hpp"
#include "uaii/ir/tensor.hpp"

#include <cstddef>
#include <cstdint>

namespace uaii {
namespace memory {

/// Host-visible tensor buffer bound to an IR tensor id.
struct TensorBuffer {
  TensorId id = 0;
  DType dtype = DType::Unknown;
  Shape shape{};
  void* data = nullptr;
  std::size_t nbytes = 0;
  bool owned = false;  // true if pool/arena owns the allocation
};

[[nodiscard]] inline float* as_f32(TensorBuffer& buf) {
  return static_cast<float*>(buf.data);
}

[[nodiscard]] inline const float* as_f32(const TensorBuffer& buf) {
  return static_cast<const float*>(buf.data);
}

}  // namespace memory
}  // namespace uaii
