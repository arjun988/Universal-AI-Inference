#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/loader.hpp"

namespace uaii {
namespace loaders {

/// MLX = safetensors weights + config.json (Apple MLX export layout) → UAII IR.
class UAII_API MlxLoader : public IModelLoader {
 public:
  [[nodiscard]] LoaderInfo info() const override;
  [[nodiscard]] bool accepts(const std::string& path) const override;
  [[nodiscard]] Error load(const std::string& path, ir::Graph* out_graph) override;
};

}  // namespace loaders
}  // namespace uaii
