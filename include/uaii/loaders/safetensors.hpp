#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/loader.hpp"
#include "uaii/interfaces/types.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace loaders {

struct SafetensorsTensorInfo {
  std::string name;
  std::string dtype;  // F32, F16, I64, …
  std::vector<std::int64_t> shape;
  std::uint64_t data_offsets[2]{0, 0};  // begin, end relative to data blob
};

struct SafetensorsFile {
  std::string path;
  std::uint64_t header_size = 0;
  std::uint64_t data_offset = 0;  // 8 + header_size
  std::unordered_map<std::string, std::string> metadata;
  std::vector<SafetensorsTensorInfo> tensors;
};

[[nodiscard]] UAII_API DType safetensors_to_dtype(const std::string& dtype) noexcept;

[[nodiscard]] UAII_API Error safetensors_read_header(const std::string& path,
                                                     SafetensorsFile* out);

[[nodiscard]] UAII_API Error safetensors_load_tensor_f32(const SafetensorsFile& file,
                                                         const std::string& tensor_name,
                                                         std::vector<float>* out,
                                                         Shape* out_shape);

/// Write a minimal F32 safetensors file (fixtures / demos).
[[nodiscard]] UAII_API Error safetensors_write_f32(
    const std::string& path,
    const std::unordered_map<std::string, std::pair<Shape, std::vector<float>>>& tensors,
    const std::unordered_map<std::string, std::string>& metadata = {});

class UAII_API SafetensorsLoader : public IModelLoader {
 public:
  [[nodiscard]] LoaderInfo info() const override;
  [[nodiscard]] bool accepts(const std::string& path) const override;
  [[nodiscard]] Error load(const std::string& path, ir::Graph* out_graph) override;
};

}  // namespace loaders
}  // namespace uaii
