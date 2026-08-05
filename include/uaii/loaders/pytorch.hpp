#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/loader.hpp"

namespace uaii {
namespace loaders {

/// PyTorch import: prefers ONNX-exported graphs beside .pt, or LibTorch when UAII_WITH_LIBTORCH.
class UAII_API PytorchLoader : public IModelLoader {
 public:
  [[nodiscard]] LoaderInfo info() const override;
  [[nodiscard]] bool accepts(const std::string& path) const override;
  [[nodiscard]] Error load(const std::string& path, ir::Graph* out_graph) override;
};

}  // namespace loaders
}  // namespace uaii
