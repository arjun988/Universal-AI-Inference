#pragma once

#include "uaii/interfaces/types.hpp"

#include <cstddef>
#include <cstdint>

namespace uaii {
namespace kernels {

struct TensorView {
  DType dtype = DType::Unknown;
  const std::int64_t* shape = nullptr;
  std::size_t rank = 0;
  void* data = nullptr;
  std::size_t nbytes = 0;

  [[nodiscard]] std::int64_t dim(std::size_t i) const {
    return i < rank ? shape[i] : 1;
  }

  [[nodiscard]] std::size_t numel() const {
    if (rank == 0) {
      return 0;
    }
    std::size_t n = 1;
    for (std::size_t i = 0; i < rank; ++i) {
      if (shape[i] < 0) {
        return 0;
      }
      n *= static_cast<std::size_t>(shape[i]);
    }
    return n;
  }

  [[nodiscard]] float* f32() { return static_cast<float*>(data); }
  [[nodiscard]] const float* f32() const { return static_cast<const float*>(data); }
};

}  // namespace kernels
}  // namespace uaii
