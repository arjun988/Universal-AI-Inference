#pragma once

#include "uaii/core/error.hpp"

#include <string>
#include <vector>

namespace uaii {

/// Opaque IR graph handle for Phase 1. Real type arrives in uaii-ir (Phase 2).
struct IrGraph;

struct LoaderInfo {
  std::string name;
  std::string version;
  std::vector<std::string> supported_extensions;
};

/// Converts an external model artifact into UAII IR.
class IModelLoader {
 public:
  virtual ~IModelLoader() = default;

  [[nodiscard]] virtual LoaderInfo info() const = 0;

  /// Returns true if this loader can handle the given path/extension.
  [[nodiscard]] virtual bool accepts(const std::string& path) const = 0;

  /// Load into an IR graph. Phase 1 stubs return NotImplemented.
  [[nodiscard]] virtual Error load(const std::string& path, IrGraph** out_graph) = 0;
};

}  // namespace uaii
