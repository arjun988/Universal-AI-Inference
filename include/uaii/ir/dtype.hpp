#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/types.hpp"
#include "uaii/ir/tensor.hpp"

#include <cstddef>
#include <string>

namespace uaii {
namespace ir {

[[nodiscard]] UAII_API bool parse_dtype(const std::string& text, DType* out) noexcept;
[[nodiscard]] UAII_API bool parse_storage_hint(const std::string& text,
                                               StorageHint* out) noexcept;

[[nodiscard]] UAII_API std::size_t dtype_size_bytes(DType dtype) noexcept;

[[nodiscard]] UAII_API std::uint64_t estimate_tensor_bytes(const Tensor& tensor) noexcept;

}  // namespace ir
}  // namespace uaii
