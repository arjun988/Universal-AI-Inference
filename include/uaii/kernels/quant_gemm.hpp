#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/kernels/tensor_view.hpp"
#include "uaii/quant/formats.hpp"

namespace uaii {
namespace kernels {

/// A (f32 [M,K]) * quantized B ([N,K] packed, transpose_b style) -> C (f32 [M,N])
[[nodiscard]] UAII_API Error quant_gemm_f32(const TensorView& a,
                                            const TensorView& b_quant,
                                            TensorView* c,
                                            bool transpose_b = true);

[[nodiscard]] UAII_API bool supports_quant_gemm(quant::QuantFormat f) noexcept;

}  // namespace kernels
}  // namespace uaii
