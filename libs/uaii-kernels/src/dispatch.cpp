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

}  // namespace

bool supports_cpu_op(const std::string& op_name, const std::string& /*op_version*/) noexcept {
  return op_name == "MatMul" || op_name == "Softmax" || op_name == "LayerNorm" ||
         op_name == "RMSNorm" || op_name == "Relu" || op_name == "Gelu" ||
         op_name == "Silu" || op_name == "Add" || op_name == "Mul" ||
         op_name == "Identity";
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

  return Error::make(ErrorCode::NotImplemented,
                     "CPU kernel not implemented for op '" + op_name + "'");
}

}  // namespace kernels
}  // namespace uaii
