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

/// tokens: f32 ids with shape [batch, seq]; weight: [vocab, dim]
[[nodiscard]] UAII_API Error embedding_f32(const TensorView& tokens,
                                           const TensorView& weight,
                                           TensorView* out);

/// Apply RoPE in-place style: out = rotate(in); optional positions [seq]
[[nodiscard]] UAII_API Error rope_f32(const TensorView& in,
                                      const TensorView* positions,
                                      TensorView* out,
                                      float theta = 10000.0f);

/// Simplified MHA: Q,K,V are [batch, seq, dim] or [batch, dim]; out same.
[[nodiscard]] UAII_API Error attention_f32(const TensorView& q,
                                           const TensorView& k,
                                           const TensorView& v,
                                           TensorView* out,
                                           int num_heads,
                                           float scale,
                                           bool causal);

/// MHA with optional past K/V ([batch, past_seq, dim]) and optional present outputs.
/// If present_* aliases past_* (same data pointer), appends new K/V in-place when capacity allows.
/// Supports rank-2 ([batch, dim] ⇒ seq=1) and rank-3 Q/K/V.
[[nodiscard]] UAII_API Error attention_kv_f32(const TensorView& q,
                                              const TensorView& k,
                                              const TensorView& v,
                                              TensorView* out,
                                              int num_heads,
                                              float scale,
                                              bool causal,
                                              const TensorView* past_k,
                                              const TensorView* past_v,
                                              TensorView* present_k,
                                              TensorView* present_v);

/// MoE router: logits = x @ gate_w^T ; probs = softmax; returns top-1 expert index in out_index (f32)
[[nodiscard]] UAII_API Error moe_router_f32(const TensorView& x,
                                            const TensorView& gate_w,
                                            TensorView* probs,
                                            TensorView* top_expert /*[batch,1] f32 ids*/);

/// MoE expert dispatch: for each row pick expert e and compute x @ We^T (experts stacked [E, dim, dim] as E separate mats via flat)
/// experts_w: [num_experts, dim, dim] flattened row-major expert-major
[[nodiscard]] UAII_API Error moe_experts_f32(const TensorView& x,
                                             const TensorView& experts_w,
                                             const TensorView& top_expert,
                                             TensorView* out,
                                             int num_experts);

[[nodiscard]] UAII_API Error reshape_f32(const TensorView& in, TensorView* out);
[[nodiscard]] UAII_API Error transpose_f32(const TensorView& in,
                                           TensorView* out,
                                           const std::vector<std::int64_t>& perm);

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
