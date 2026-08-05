#pragma once

#include "uaii/core/error.hpp"
#include "uaii/ir/dtype.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <string>

namespace uaii {
namespace kernels {

/// Ensure logical shape fits in nbytes for the declared dtype.
[[nodiscard]] inline Error check_view_bytes(const TensorView& t, const char* name) {
  if (t.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, std::string(name) + " data null");
  }
  const std::size_t elem = ir::dtype_size_bytes(t.dtype);
  if (elem == 0) {
    return Error::make(ErrorCode::InvalidArgument, std::string(name) + " unknown dtype");
  }
  const std::size_t need = t.numel() * elem;
  if (need == 0) {
    return Error::make(ErrorCode::InvalidArgument, std::string(name) + " empty numel");
  }
  if (t.nbytes < need) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string(name) + " nbytes " + std::to_string(t.nbytes) +
                           " < required " + std::to_string(need));
  }
  return Error::success();
}

}  // namespace kernels
}  // namespace uaii
