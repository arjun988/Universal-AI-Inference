#pragma once

#include "uaii/interfaces/types.hpp"

#include <cstdint>
#include <string>

namespace uaii {
namespace quant {

enum class QuantFormat {
  F32 = 0,
  F16,
  BF16,
  INT8,
  INT4,
  NF4,
  MXFP4,
};

[[nodiscard]] inline const char* to_string(QuantFormat f) noexcept {
  switch (f) {
    case QuantFormat::F32: return "f32";
    case QuantFormat::F16: return "f16";
    case QuantFormat::BF16: return "bf16";
    case QuantFormat::INT8: return "int8";
    case QuantFormat::INT4: return "int4";
    case QuantFormat::NF4: return "nf4";
    case QuantFormat::MXFP4: return "mxfp4";
    default: return "unknown";
  }
}

[[nodiscard]] inline bool parse_quant_format(const std::string& text, QuantFormat* out) noexcept {
  if (out == nullptr) return false;
  if (text == "f32" || text == "fp32") { *out = QuantFormat::F32; return true; }
  if (text == "f16" || text == "fp16") { *out = QuantFormat::F16; return true; }
  if (text == "bf16") { *out = QuantFormat::BF16; return true; }
  if (text == "int8" || text == "i8") { *out = QuantFormat::INT8; return true; }
  if (text == "int4" || text == "i4") { *out = QuantFormat::INT4; return true; }
  if (text == "nf4") { *out = QuantFormat::NF4; return true; }
  if (text == "mxfp4") { *out = QuantFormat::MXFP4; return true; }
  return false;
}

/// Packed storage bytes for `n` elements (excluding scale metadata).
[[nodiscard]] inline std::size_t packed_nbytes(QuantFormat f, std::size_t n) noexcept {
  switch (f) {
    case QuantFormat::F32: return n * 4;
    case QuantFormat::F16:
    case QuantFormat::BF16: return n * 2;
    case QuantFormat::INT8: return n;
    case QuantFormat::INT4:
    case QuantFormat::NF4:
    case QuantFormat::MXFP4: return (n + 1) / 2;
    default: return 0;
  }
}

}  // namespace quant
}  // namespace uaii
