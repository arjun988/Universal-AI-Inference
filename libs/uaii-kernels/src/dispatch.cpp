#include "uaii/kernels/kernels.hpp"

namespace uaii {
namespace kernels {
namespace {

bool attr_bool(const std::vector<ir::Attribute>& attrs, const char* key, bool def) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::Bool) {
      return std::get<bool>(a.value);
    }
  }
  return def;
}

std::int64_t attr_int(const std::vector<ir::Attribute>& attrs, const char* key,
                      std::int64_t def) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::Int) {
      return std::get<std::int64_t>(a.value);
    }
  }
  return def;
}

float attr_float(const std::vector<ir::Attribute>& attrs, const char* key, float def) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::Float) {
      return static_cast<float>(std::get<double>(a.value));
    }
    if (a.key == key && a.type == ir::AttributeType::Int) {
      return static_cast<float>(std::get<std::int64_t>(a.value));
    }
  }
  return def;
}

std::vector<std::int64_t> attr_int_array(const std::vector<ir::Attribute>& attrs,
                                         const char* key) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::IntArray) {
      return std::get<std::vector<std::int64_t>>(a.value);
    }
  }
  return {};
}

}  // namespace

bool supports_cpu_op(const std::string& op_name, const std::string& /*op_version*/) noexcept {
  return op_name == "MatMul" || op_name == "Softmax" || op_name == "LayerNorm" ||
         op_name == "RMSNorm" || op_name == "Relu" || op_name == "Gelu" ||
         op_name == "Silu" || op_name == "Add" || op_name == "Mul" ||
         op_name == "Identity" || op_name == "Embedding" || op_name == "RoPE" ||
         op_name == "Attention" || op_name == "MoERouter" || op_name == "MoEExperts" ||
         op_name == "Reshape" || op_name == "Transpose" || op_name == "MLP" ;
}

Error dispatch_cpu(const std::string& op_name,
                   const std::string& /*op_version*/,
                   const std::vector<TensorView>& inputs,
                   std::vector<TensorView>* outputs,
                   const std::vector<ir::Attribute>& attrs) {
  if (outputs == nullptr || outputs->empty()) {
    return Error::make(ErrorCode::InvalidArgument, "dispatch outputs empty");
  }

  if (op_name == "MatMul") {
    if (inputs.size() != 2 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "MatMul expects 2 inputs / 1 output");
    }
    return matmul_f32(inputs[0], inputs[1], &(*outputs)[0],
                      attr_bool(attrs, "transpose_a", false),
                      attr_bool(attrs, "transpose_b", false));
  }
  if (op_name == "Softmax") {
    if (inputs.size() != 1 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Softmax expects 1 in / 1 out");
    }
    return softmax_f32(inputs[0], &(*outputs)[0],
                       static_cast<int>(attr_int(attrs, "axis", -1)));
  }
  if (op_name == "LayerNorm") {
    if (inputs.empty() || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "LayerNorm arity");
    }
    const TensorView* w = inputs.size() > 1 ? &inputs[1] : nullptr;
    const TensorView* b = inputs.size() > 2 ? &inputs[2] : nullptr;
    return layernorm_f32(inputs[0], w, b, &(*outputs)[0],
                         attr_float(attrs, "eps", 1e-5f));
  }
  if (op_name == "RMSNorm") {
    if (inputs.empty() || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "RMSNorm arity");
    }
    const TensorView* w = inputs.size() > 1 ? &inputs[1] : nullptr;
    return rmsnorm_f32(inputs[0], w, &(*outputs)[0], attr_float(attrs, "eps", 1e-5f));
  }
  if (op_name == "Relu") {
    if (inputs.size() != 1 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Relu arity");
    }
    return relu_f32(inputs[0], &(*outputs)[0]);
  }
  if (op_name == "Gelu") {
    if (inputs.size() != 1 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Gelu arity");
    }
    return gelu_f32(inputs[0], &(*outputs)[0]);
  }
  if (op_name == "Silu") {
    if (inputs.size() != 1 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Silu arity");
    }
    return silu_f32(inputs[0], &(*outputs)[0]);
  }
  if (op_name == "Add") {
    if (inputs.size() != 2 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Add arity");
    }
    return add_f32(inputs[0], inputs[1], &(*outputs)[0]);
  }
  if (op_name == "Mul") {
    if (inputs.size() != 2 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Mul arity");
    }
    return mul_f32(inputs[0], inputs[1], &(*outputs)[0]);
  }
  if (op_name == "Identity") {
    if (inputs.size() != 1 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Identity arity");
    }
    return identity_f32(inputs[0], &(*outputs)[0]);
  }
  if (op_name == "Embedding") {
    if (inputs.size() != 2 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Embedding arity");
    }
    return embedding_f32(inputs[0], inputs[1], &(*outputs)[0]);
  }
  if (op_name == "RoPE") {
    if (inputs.empty() || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "RoPE arity");
    }
    const TensorView* pos = inputs.size() > 1 ? &inputs[1] : nullptr;
    return rope_f32(inputs[0], pos, &(*outputs)[0], attr_float(attrs, "theta", 10000.f));
  }
  if (op_name == "Attention") {
    if (inputs.size() < 3 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Attention expects Q,K,V");
    }
    return attention_f32(inputs[0], inputs[1], inputs[2], &(*outputs)[0],
                         static_cast<int>(attr_int(attrs, "num_heads", 1)),
                         attr_float(attrs, "scale", 0.f),
                         attr_bool(attrs, "causal", true));
  }
  if (op_name == "MoERouter") {
    if (inputs.size() != 2 || outputs->size() < 2) {
      return Error::make(ErrorCode::InvalidArgument,
                         "MoERouter expects x,gate -> probs,top_expert");
    }
    return moe_router_f32(inputs[0], inputs[1], &(*outputs)[0], &(*outputs)[1]);
  }
  if (op_name == "MoEExperts") {
    if (inputs.size() != 3 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument,
                         "MoEExperts expects x,experts_w,top_expert");
    }
    return moe_experts_f32(inputs[0], inputs[1], inputs[2], &(*outputs)[0],
                           static_cast<int>(attr_int(attrs, "num_experts", 1)));
  }
  if (op_name == "Reshape") {
    if (inputs.size() != 1 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Reshape arity");
    }
    return reshape_f32(inputs[0], &(*outputs)[0]);
  }
  if (op_name == "Transpose") {
    if (inputs.size() != 1 || outputs->size() != 1) {
      return Error::make(ErrorCode::InvalidArgument, "Transpose arity");
    }
    auto perm = attr_int_array(attrs, "perm");
    if (perm.empty()) perm = {1, 0};
    return transpose_f32(inputs[0], &(*outputs)[0], perm);
  }
  if (op_name == "MLP") {
    // Convenience: x, w1, w2 -> Gelu(x@w1)@w2  (needs temp — not available).
    // Expand as MatMul/Gelu/MatMul in graphs; keep schema for loaders.
    return Error::make(ErrorCode::NotImplemented,
                       "MLP is a composite op; expand to MatMul/Gelu/MatMul in IR");
  }

  return Error::make(ErrorCode::NotImplemented,
                     "CPU kernel not implemented for op '" + op_name + "'");
}

}  // namespace kernels
}  // namespace uaii
