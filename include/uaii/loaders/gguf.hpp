#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/loader.hpp"
#include "uaii/interfaces/types.hpp"
#include "uaii/quant/formats.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace uaii {
namespace loaders {

enum class GgufType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  I8 = 16,
  I16 = 17,
  I32 = 18,
  I64 = 19,
  F64 = 20,
};

struct GgufTensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;  // GGUF stores dims reversed vs row-major often [n,m]
  GgufType type = GgufType::F32;
  std::uint64_t offset = 0;  // relative to data section
};

using GgufValue = std::variant<std::monostate,
                               bool,
                               std::uint8_t,
                               std::int8_t,
                               std::uint16_t,
                               std::int16_t,
                               std::uint32_t,
                               std::int32_t,
                               std::uint64_t,
                               std::int64_t,
                               float,
                               double,
                               std::string,
                               std::vector<std::string>,
                               std::vector<std::int32_t>,
                               std::vector<std::int64_t>,
                               std::vector<float>>;

struct GgufFile {
  std::uint32_t version = 0;
  std::unordered_map<std::string, GgufValue> kv;
  std::vector<GgufTensorInfo> tensors;
  std::uint64_t data_offset = 0;  // absolute file offset of first tensor byte
  std::string path;
};

[[nodiscard]] UAII_API const char* to_string(GgufType type) noexcept;
[[nodiscard]] UAII_API DType gguf_to_dtype(GgufType type) noexcept;
[[nodiscard]] UAII_API std::size_t gguf_type_nbytes_per_elem(GgufType type) noexcept;

[[nodiscard]] UAII_API Error gguf_read_header(const std::string& path, GgufFile* out);
[[nodiscard]] UAII_API Error gguf_load_tensor_f32(const GgufFile& file,
                                                  const std::string& tensor_name,
                                                  std::vector<float>* out,
                                                  Shape* out_shape);

/// Load raw packed bytes (no dequant) for in-memory quant GEMM.
[[nodiscard]] UAII_API Error gguf_load_tensor_raw(const GgufFile& file,
                                                  const std::string& tensor_name,
                                                  std::vector<std::uint8_t>* out,
                                                  Shape* out_shape,
                                                  GgufType* out_type);

/// True for F32/F16 and GGUF block quants supported on the transformer import path.
[[nodiscard]] UAII_API bool gguf_type_supported(GgufType t) noexcept;

[[nodiscard]] UAII_API quant::QuantFormat gguf_type_to_quant(GgufType t) noexcept;

/// Write a minimal F32-only GGUF (for fixtures / demos).
[[nodiscard]] UAII_API Error gguf_write_f32(
    const std::string& path,
    const std::unordered_map<std::string, GgufValue>& kv,
    const std::vector<std::pair<GgufTensorInfo, std::vector<float>>>& tensors);

class UAII_API GgufLoader : public IModelLoader {
 public:
  [[nodiscard]] LoaderInfo info() const override;
  [[nodiscard]] bool accepts(const std::string& path) const override;
  [[nodiscard]] Error load(const std::string& path, ir::Graph* out_graph) override;
};

}  // namespace loaders
}  // namespace uaii
