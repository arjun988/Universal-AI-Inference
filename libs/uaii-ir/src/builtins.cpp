#include "uaii/ir/registry.hpp"

namespace uaii {
namespace ir {
namespace {

OpSchema schema(std::string name,
                int min_in,
                int max_in,
                int min_out,
                int max_out,
                std::vector<std::string> attrs,
                std::string desc) {
  OpSchema s;
  s.name = std::move(name);
  s.version = "1";
  s.min_inputs = min_in;
  s.max_inputs = max_in;
  s.min_outputs = min_out;
  s.max_outputs = max_out;
  s.known_attributes = std::move(attrs);
  s.description = std::move(desc);
  return s;
}

}  // namespace

void OperatorRegistry::register_builtin_schemas() {
  // Linear algebra / elementwise
  (void)register_schema(schema("MatMul", 2, 2, 1, 1, {"transpose_a", "transpose_b"},
                               "Matrix multiplication"));
  (void)register_schema(schema("MatMulRelu", 2, 2, 1, 1, {"transpose_a", "transpose_b"},
                               "Fused MatMul + ReLU (Phase 6)"));
  (void)register_schema(schema("Add", 2, 2, 1, 1, {}, "Element-wise add"));
  (void)register_schema(schema("Mul", 2, 2, 1, 1, {}, "Element-wise multiply"));
  (void)register_schema(schema("Sub", 2, 2, 1, 1, {}, "Element-wise subtract"));
  (void)register_schema(schema("Div", 2, 2, 1, 1, {}, "Element-wise divide"));

  // Activations / norms
  (void)register_schema(schema("Relu", 1, 1, 1, 1, {}, "ReLU"));
  (void)register_schema(schema("Neg", 1, 1, 1, 1, {}, "Negate (plugin or builtin)"));
  (void)register_schema(schema("Gelu", 1, 1, 1, 1, {"approximate"}, "GELU"));
  (void)register_schema(schema("Silu", 1, 1, 1, 1, {}, "SiLU / Swish"));
  (void)register_schema(schema("Softmax", 1, 1, 1, 1, {"axis"}, "Softmax"));
  (void)register_schema(schema("LayerNorm", 1, 3, 1, 1, {"eps", "axis"}, "Layer normalization"));
  (void)register_schema(schema("RMSNorm", 1, 2, 1, 1, {"eps"}, "RMS normalization"));

  // Tensor ops
  (void)register_schema(schema("Reshape", 1, 2, 1, 1, {"shape"}, "Reshape"));
  (void)register_schema(schema("Transpose", 1, 1, 1, 1, {"perm"}, "Transpose"));
  (void)register_schema(schema("Identity", 1, 1, 1, 1, {}, "Identity"));
  (void)register_schema(schema("Cast", 1, 1, 1, 1, {"to"}, "Cast dtype"));
  (void)register_schema(schema("Concat", 1, -1, 1, 1, {"axis"}, "Concatenate"));
  (void)register_schema(schema("Split", 1, 1, 1, -1, {"axis", "split"}, "Split"));

  // Attention / LLM building blocks (Phase 4)
  (void)register_schema(schema("Attention", 3, 5, 1, 3,
                               {"num_heads", "kv_heads", "scale", "causal",
                                "use_kv_cache", "layer_id"},
                               "Multi-head attention (Q,K,V[,past_k,past_v])"));
  (void)register_schema(schema("RoPE", 1, 2, 1, 1, {"theta", "interleaved"},
                               "Rotary position embedding"));
  (void)register_schema(schema("Embedding", 2, 2, 1, 1, {}, "Token embedding lookup"));
  (void)register_schema(schema("MLP", 3, 3, 1, 1, {"activation"},
                               "Composite MLP (expand in IR for execution)"));

  // MoE
  (void)register_schema(schema("MoERouter", 2, 2, 2, 2, {"num_experts", "top_k"},
                               "MoE gate router → probs + top expert ids"));
  (void)register_schema(schema("MoEExperts", 3, 3, 1, 1, {"num_experts"},
                               "MoE expert dispatch (top-1)"));
}

}  // namespace ir
}  // namespace uaii
