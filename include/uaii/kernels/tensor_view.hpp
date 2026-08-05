#pragma once

#include "uaii/interfaces/types.hpp"
#include "uaii/quant/formats.hpp"

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

  /// When set, `data` holds packed quantized weights (GGUF block layout).
  quant::QuantFormat quant_format = quant::QuantFormat::F32;
  std::int64_t quant_rows = 0;  // logical rows (e.g. out_features)
  std::int64_t quant_cols = 0;  // logical cols (e.g. in_features)

  [[nodiscard]] bool is_quantized() const noexcept {
    return quant::is_gguf_block_quant(quant_format) ||
           (quant_format != quant::QuantFormat::F32 && quant_format != quant::QuantFormat::F16 &&
            quant_format != quant::QuantFormat::BF16 && dtype != DType::F32);
  }

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
