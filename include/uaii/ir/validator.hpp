#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/registry.hpp"

#include <string>
#include <vector>

namespace uaii {
namespace ir {

enum class ValidationSeverity {
  Error,
  Warning,
};

struct ValidationIssue {
  ValidationSeverity severity = ValidationSeverity::Error;
  std::string code;
  std::string message;
  NodeId node_id = 0;
  TensorId tensor_id = 0;
};

struct ValidationOptions {
  bool allow_unknown_ops = false;
  bool require_shapes = true;
  bool require_dtypes = true;
  bool check_cycles = true;
};

struct ValidationResult {
  std::vector<ValidationIssue> issues;

  [[nodiscard]] bool ok() const noexcept {
    for (const auto& i : issues) {
      if (i.severity == ValidationSeverity::Error) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::size_t error_count() const noexcept {
    std::size_t n = 0;
    for (const auto& i : issues) {
      if (i.severity == ValidationSeverity::Error) {
        ++n;
      }
    }
    return n;
  }
};

[[nodiscard]] UAII_API ValidationResult validate_graph(
    const Graph& graph,
    const OperatorRegistry& registry,
    const ValidationOptions& options = {});

/// Convenience: Error::ok on success, ConfigError-style message on failure.
[[nodiscard]] UAII_API Error validate_graph_error(
    const Graph& graph,
    const OperatorRegistry& registry,
    const ValidationOptions& options = {});

}  // namespace ir
}  // namespace uaii
