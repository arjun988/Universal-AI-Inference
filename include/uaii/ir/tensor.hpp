#pragma once

#include "uaii/interfaces/types.hpp"
#include "uaii/quant/formats.hpp"

#include <cstddef>
#include <string>

namespace uaii {
namespace ir {

enum class StorageHint {
  Unspecified = 0,
  Ram,
  Mmap,
  External,
};

/// A typed tensor value in the IR graph (activation, input, or weight ref).
struct Tensor {
  TensorId id = 0;
  std::string name;
  DType dtype = DType::Unknown;
  Shape shape{};
  StorageHint storage_hint = StorageHint::Unspecified;
  bool is_weight = false;
  /// Optional external weight locator (file URI, pack key, …).
  std::string weight_ref;
  /// Packed weight format for in-memory quant GEMM (F32 = dense).
  quant::QuantFormat quant_format = quant::QuantFormat::F32;
};

[[nodiscard]] inline const char* to_string(StorageHint hint) noexcept {
  switch (hint) {
    case StorageHint::Ram: return "ram";
    case StorageHint::Mmap: return "mmap";
    case StorageHint::External: return "external";
    default: return "unspecified";
  }
}

[[nodiscard]] inline std::size_t shape_numel(const Shape& shape) {
  if (shape.dims.empty()) {
    return 0;
  }
  std::size_t n = 1;
  for (std::int64_t d : shape.dims) {
    if (d < 0) {
      return 0;  // dynamic
    }
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

}  // namespace ir
}  // namespace uaii
