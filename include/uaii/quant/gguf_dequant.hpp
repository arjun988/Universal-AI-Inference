#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/quant/formats.hpp"

#include <cstdint>

namespace uaii {
namespace quant {

/// Dequantize one GGUF/ggml row (Q4_0 … Q6_K) into `cols` f32 values.
[[nodiscard]] UAII_API Error dequant_gguf_row(QuantFormat format,
                                              const std::uint8_t* packed,
                                              std::int64_t cols,
                                              float* out);

[[nodiscard]] inline bool supports_gguf_dequant(QuantFormat f) noexcept {
  return is_gguf_block_quant(f);
}

}  // namespace quant
}  // namespace uaii
