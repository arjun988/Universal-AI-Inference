#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <string>
#include <vector>

namespace uaii {
namespace kernels {

[[nodiscard]] UAII_API Error matmul_f32(const TensorView& a,
                                        const TensorView& b,
                                        TensorView* c,
                                        bool transpose_a = false,
                                        bool transpose_b = false);

[[nodiscard]] UAII_API Error softmax_f32(const TensorView& in,
                                         TensorView* out,
                                         int axis = -1);

[[nodiscard]] UAII_API Error layernorm_f32(const TensorView& in,
                                           const TensorView* weight,
                                           const TensorView* bias,
                                           TensorView* out,
                                           float eps = 1e-5f);

[[nodiscard]] UAII_API Error rmsnorm_f32(const TensorView& in,
                                         const TensorView* weight,
                                         TensorView* out,
                                         float eps = 1e-5f);

[[nodiscard]] UAII_API Error relu_f32(const TensorView& in, TensorView* out);
[[nodiscard]] UAII_API Error gelu_f32(const TensorView& in, TensorView* out);
[[nodiscard]] UAII_API Error silu_f32(const TensorView& in, TensorView* out);

[[nodiscard]] UAII_API Error add_f32(const TensorView& a,
                                     const TensorView& b,
                                     TensorView* out);
[[nodiscard]] UAII_API Error mul_f32(const TensorView& a,
                                     const TensorView& b,
                                     TensorView* out);

[[nodiscard]] UAII_API Error identity_f32(const TensorView& in, TensorView* out);

/// Dispatch by operator name for CPU execution.
[[nodiscard]] UAII_API Error dispatch_cpu(const std::string& op_name,
                                          const std::string& op_version,
                                          const std::vector<TensorView>& inputs,
                                          std::vector<TensorView>* outputs,
                                          const std::vector<ir::Attribute>& attrs);

[[nodiscard]] UAII_API bool supports_cpu_op(const std::string& op_name,
                                            const std::string& op_version) noexcept;

}  // namespace kernels
}  // namespace uaii
