#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace uaii {

using TensorId = std::uint64_t;
using NodeId = std::uint64_t;
using DeviceId = std::uint32_t;

enum class DType {
  Unknown = 0,
  F32,
  F16,
  BF16,
  I8,
  I32,
  I64,
  U8,
  U32,
  Bool,
};

enum class DeviceType {
  Unknown = 0,
  Cpu,
  Cuda,
  Metal,
  Vulkan,
  WebGpu,
  Rocm,
  Npu,
  Custom,
};

struct Shape {
  std::vector<std::int64_t> dims;
};

struct TensorDesc {
  TensorId id = 0;
  DType dtype = DType::Unknown;
  Shape shape{};
  std::string name;
};

[[nodiscard]] inline const char* to_string(DType dtype) noexcept {
  switch (dtype) {
    case DType::F32: return "f32";
    case DType::F16: return "f16";
    case DType::BF16: return "bf16";
    case DType::I8: return "i8";
    case DType::I32: return "i32";
    case DType::I64: return "i64";
    case DType::U8: return "u8";
    case DType::U32: return "u32";
    case DType::Bool: return "bool";
    default: return "unknown";
  }
}

[[nodiscard]] inline const char* to_string(DeviceType type) noexcept {
  switch (type) {
    case DeviceType::Cpu: return "cpu";
    case DeviceType::Cuda: return "cuda";
    case DeviceType::Metal: return "metal";
    case DeviceType::Vulkan: return "vulkan";
    case DeviceType::WebGpu: return "webgpu";
    case DeviceType::Rocm: return "rocm";
    case DeviceType::Npu: return "npu";
    case DeviceType::Custom: return "custom";
    default: return "unknown";
  }
}

}  // namespace uaii
