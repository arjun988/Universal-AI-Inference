#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/loader.hpp"

namespace uaii {
namespace loaders {

/// Minimal ONNX → UAII IR importer (protobuf text/binary subset + common MLP/transformer ops).
class UAII_API OnnxLoader : public IModelLoader {
 public:
  [[nodiscard]] LoaderInfo info() const override;
  [[nodiscard]] bool accepts(const std::string& path) const override;
  [[nodiscard]] Error load(const std::string& path, ir::Graph* out_graph) override;
};

}  // namespace loaders
}  // namespace uaii
