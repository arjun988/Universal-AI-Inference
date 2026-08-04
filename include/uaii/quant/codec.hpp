#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/quant/formats.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace uaii {
namespace quant {

struct QuantParams {
  float scale = 1.f;
  float zero_point = 0.f;
  /// Group size for NF4/MXFP4-style blocked scales (0 = per-tensor).
  int group_size = 0;
};

/// Pack f32 → quantized bytes (+ optional per-group scales for NF4/MXFP4).
[[nodiscard]] UAII_API Error pack_f32(const float* src,
                                      std::size_t n,
                                      QuantFormat format,
                                      const QuantParams& params,
                                      std::vector<std::uint8_t>* packed,
                                      std::vector<float>* scales);

/// Unpack quantized bytes → f32.
[[nodiscard]] UAII_API Error unpack_to_f32(const std::uint8_t* packed,
                                           std::size_t packed_nbytes,
                                           std::size_t n,
                                           QuantFormat format,
                                           const QuantParams& params,
                                           const float* scales,
                                           std::size_t n_scales,
                                           float* dst);

}  // namespace quant
}  // namespace uaii
