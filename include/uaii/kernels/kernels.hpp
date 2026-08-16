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

/// Apply RoPE: out = rotate(in). Optional positions [rows].
/// When head_dim > 0, RoPE is applied per head chunk (required for MHA/GQA).
/// When rotary_dim > 0 and rotary_dim < head_dim, only the trailing rotary_dim
/// elements of each head are rotated (DeepSeek MLA).
/// When positions is null and a KV cache is active, uses cache past_len as position
/// for each row (seq=1 decode) or past_len+row for multi-row tensors.
/// mode: 0 = interleaved pairs (i, i+1); 2 = NeoX pairs (i, i+head_dim/2)
/// (llama.cpp convention; Qwen2/Phi/Falcon need mode=2).
/// scale_type: empty/none/default = no scaling; linear; llama3; dynamic; yarn.
[[nodiscard]] UAII_API Error rope_f32(const TensorView& in,
                                      const TensorView* positions,
                                      TensorView* out,
                                      float theta = 10000.0f,
                                      std::int64_t head_dim = 0,
                                      int mode = 0,
                                      std::int64_t rotary_dim = 0,
                                      const char* scale_type = nullptr,
                                      float scale_factor = 1.f,
                                      float low_freq_factor = 1.f,
                                      float high_freq_factor = 4.f,
                                      float old_context_len = 8192.f,
                                      float attn_factor = 1.f);

/// Inverse frequency after Hugging Face rope_scaling (pair index 0 .. n_pairs-1).
[[nodiscard]] UAII_API float rope_scaled_inv_freq(std::int64_t pair_index,
                                                  std::int64_t n_pairs,
                                                  float theta,
                                                  const char* scale_type,
                                                  float scale_factor,
                                                  float low_freq_factor,
                                                  float high_freq_factor,
                                                  float old_context_len);

/// Assemble MLA K: per head concat(k_nope, broadcast(k_rope)).
/// k_nope: [B, H*nope], k_rope: [B, rope], out: [B, H*(nope+rope)].
[[nodiscard]] UAII_API Error mla_expand_k_f32(const TensorView& k_nope,
                                              const TensorView& k_rope,
                                              TensorView* out,
                                              int num_heads,
                                              std::int64_t nope_dim,
                                              std::int64_t rope_dim);

/// Simplified MHA: Q,K,V are [batch, seq, dim] or [batch, dim]; out same.
[[nodiscard]] UAII_API Error attention_f32(const TensorView& q,
                                           const TensorView& k,
                                           const TensorView& v,
                                           TensorView* out,
                                           int num_heads,
                                           float scale,
                                           bool causal,
                                           float logit_softcap = 0.f);

/// MHA with optional past K/V. logit_softcap > 0 applies Gemma-2-style tanh capping.
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
                                              TensorView* present_v,
                                              float logit_softcap = 0.f);

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

/// Mixtral/Qwen2MoE-style SwiGLU experts with top-k weighted merge.
/// gate/up: [E, I, D], down: [E, D, I] (row-major, matches UAII GGUF dims_to_shape).
/// probs: [B, E] from MoERouter. top_k<=0 → 1.
[[nodiscard]] UAII_API Error moe_experts_swiglu_f32(const TensorView& x,
                                                    const TensorView& gate_exps,
                                                    const TensorView& up_exps,
                                                    const TensorView& down_exps,
                                                    const TensorView& probs,
                                                    TensorView* out,
                                                    int num_experts,
                                                    int top_k);

[[nodiscard]] UAII_API Error moe_experts_swiglu_listed_f32(const TensorView& x,
                                                           const TensorView& probs,
                                                           const TensorView* expert_triples,
                                                           int num_experts,
                                                           int top_k,
                                                           TensorView* out);

[[nodiscard]] UAII_API Error reshape_f32(const TensorView& in, TensorView* out);

/// Slice contiguous range [start, end) on the last dimension (rank-2 [B,D] only).
[[nodiscard]] UAII_API Error slice_f32(const TensorView& in,
                                       TensorView* out,
                                       std::int64_t start,
                                       std::int64_t end);
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
