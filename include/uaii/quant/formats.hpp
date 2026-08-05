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
  // GGUF block formats (in-memory compute)
  Q4_0,
  Q4_1,
  Q5_0,
  Q5_1,
  Q8_0,
  Q2_K,
  Q3_K,
  Q4_K,
  Q5_K,
  Q6_K,
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
    case QuantFormat::Q4_0: return "q4_0";
    case QuantFormat::Q4_1: return "q4_1";
    case QuantFormat::Q5_0: return "q5_0";
    case QuantFormat::Q5_1: return "q5_1";
    case QuantFormat::Q8_0: return "q8_0";
    case QuantFormat::Q2_K: return "q2_k";
    case QuantFormat::Q3_K: return "q3_k";
    case QuantFormat::Q4_K: return "q4_k";
    case QuantFormat::Q5_K: return "q5_k";
    case QuantFormat::Q6_K: return "q6_k";
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
  if (text == "q4_0") { *out = QuantFormat::Q4_0; return true; }
  if (text == "q4_1") { *out = QuantFormat::Q4_1; return true; }
  if (text == "q5_0") { *out = QuantFormat::Q5_0; return true; }
  if (text == "q5_1") { *out = QuantFormat::Q5_1; return true; }
  if (text == "q8_0") { *out = QuantFormat::Q8_0; return true; }
  if (text == "q2_k") { *out = QuantFormat::Q2_K; return true; }
  if (text == "q3_k") { *out = QuantFormat::Q3_K; return true; }
  if (text == "q4_k") { *out = QuantFormat::Q4_K; return true; }
  if (text == "q5_k") { *out = QuantFormat::Q5_K; return true; }
  if (text == "q6_k") { *out = QuantFormat::Q6_K; return true; }
  return false;
}

[[nodiscard]] inline bool is_gguf_block_quant(QuantFormat f) noexcept {
  return f >= QuantFormat::Q4_0 && f <= QuantFormat::Q6_K;
}

[[nodiscard]] inline std::size_t packed_nbytes(QuantFormat f, std::size_t n) noexcept {
  switch (f) {
    case QuantFormat::F32: return n * 4;
    case QuantFormat::F16:
    case QuantFormat::BF16: return n * 2;
    case QuantFormat::INT8: return n;
    case QuantFormat::INT4:
    case QuantFormat::NF4:
    case QuantFormat::MXFP4: return (n + 1) / 2;
    case QuantFormat::Q4_0: return (n / 32) * 18;   // f16 + 16 bytes
    case QuantFormat::Q4_1: return (n / 32) * 20;
    case QuantFormat::Q5_0: return (n / 32) * 22;
    case QuantFormat::Q5_1: return (n / 32) * 24;
    case QuantFormat::Q8_0: return (n / 32) * 34;   // f16 + 32 i8
    case QuantFormat::Q2_K: return (n / 256) * 84;
    case QuantFormat::Q3_K: return (n / 256) * 110;
    case QuantFormat::Q4_K: return (n / 256) * 144;
    case QuantFormat::Q5_K: return (n / 256) * 176;
    case QuantFormat::Q6_K: return (n / 256) * 210;
    default: return 0;
  }
}

}  // namespace quant
}  // namespace uaii
