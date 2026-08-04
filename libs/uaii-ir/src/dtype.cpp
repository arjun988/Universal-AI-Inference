#include "uaii/ir/dtype.hpp"

#include <cctype>

namespace uaii {
namespace ir {
namespace {

std::string lower(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

}  // namespace

bool parse_dtype(const std::string& text, DType* out) noexcept {
  if (out == nullptr) {
    return false;
  }
  const std::string t = lower(text);
  if (t == "f32" || t == "float32" || t == "float") { *out = DType::F32; return true; }
  if (t == "f16" || t == "float16" || t == "half") { *out = DType::F16; return true; }
  if (t == "bf16" || t == "bfloat16") { *out = DType::BF16; return true; }
  if (t == "i8" || t == "int8") { *out = DType::I8; return true; }
  if (t == "i32" || t == "int32") { *out = DType::I32; return true; }
  if (t == "i64" || t == "int64") { *out = DType::I64; return true; }
  if (t == "u8" || t == "uint8") { *out = DType::U8; return true; }
  if (t == "u32" || t == "uint32") { *out = DType::U32; return true; }
  if (t == "bool" || t == "boolean") { *out = DType::Bool; return true; }
  if (t == "unknown") { *out = DType::Unknown; return true; }
  return false;
}

bool parse_storage_hint(const std::string& text, StorageHint* out) noexcept {
  if (out == nullptr) {
    return false;
  }
  const std::string t = lower(text);
  if (t == "unspecified" || t == "default" || t.empty()) {
    *out = StorageHint::Unspecified;
    return true;
  }
  if (t == "ram") { *out = StorageHint::Ram; return true; }
  if (t == "mmap") { *out = StorageHint::Mmap; return true; }
  if (t == "external") { *out = StorageHint::External; return true; }
  return false;
}

std::size_t dtype_size_bytes(DType dtype) noexcept {
  switch (dtype) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::I8: return 1;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::U8: return 1;
    case DType::U32: return 4;
    case DType::Bool: return 1;
    default: return 0;
  }
}

std::uint64_t estimate_tensor_bytes(const Tensor& tensor) noexcept {
  const std::size_t elem = dtype_size_bytes(tensor.dtype);
  const std::size_t n = shape_numel(tensor.shape);
  if (elem == 0 || n == 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(elem) * static_cast<std::uint64_t>(n);
}

}  // namespace ir
}  // namespace uaii
