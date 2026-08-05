#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace uaii {
namespace loaders {

class UAII_API LoaderRegistry {
 public:
  void register_loader(std::unique_ptr<IModelLoader> loader);
  [[nodiscard]] IModelLoader* find_for_path(const std::string& path) const;
  [[nodiscard]] std::vector<LoaderInfo> list() const;

  void register_defaults();

 private:
  std::vector<std::unique_ptr<IModelLoader>> loaders_;
};

[[nodiscard]] UAII_API LoaderRegistry& default_loaders();

/// Load any supported model format into UAII IR.
[[nodiscard]] UAII_API Error load_model(const std::string& path, ir::Graph* out);

/// Convert model file → UAII IR file (json/binary by extension).
[[nodiscard]] UAII_API Error convert_model(const std::string& input_path,
                                           const std::string& output_path);

/// Resolve weight_ref of the form path#tensor or raw .bin into f32 bytes.
[[nodiscard]] UAII_API Error load_weight_ref_f32(const std::string& weight_ref,
                                                 const std::string& weights_dir,
                                                 const Shape& expected_shape,
                                                 float* dst,
                                                 std::size_t nbytes);

/// Load packed quant bytes when available; sets *out_format. Falls back to f32 unpack
/// into dst_f32 if keep_packed is false or format is dense.
[[nodiscard]] UAII_API Error load_weight_ref_auto(const std::string& weight_ref,
                                                  const std::string& weights_dir,
                                                  const Shape& expected_shape,
                                                  bool keep_packed,
                                                  std::vector<std::uint8_t>* packed_out,
                                                  quant::QuantFormat* out_format,
                                                  float* dst_f32,
                                                  std::size_t nbytes_f32);

}  // namespace loaders
}  // namespace uaii
