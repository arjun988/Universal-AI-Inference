#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/plan.hpp"

#include <string>

namespace uaii {
namespace ir {

enum class IrFormat {
  Auto = 0,
  Json,
  Binary,
};

/// Magic for native binary IR files: 'U''A''I''R'
inline constexpr char kBinaryMagic[4] = {'U', 'A', 'I', 'R'};

[[nodiscard]] UAII_API const char* to_string(IrFormat format) noexcept;

/// Infer format from path extension (.json / .uaii.json → Json, .uaii → Binary).
[[nodiscard]] UAII_API IrFormat infer_format_from_path(const std::string& path) noexcept;

// --- JSON (debug / hand-authored) -------------------------------------------
[[nodiscard]] UAII_API Error graph_to_json(const Graph& graph, std::string* out);
[[nodiscard]] UAII_API Error graph_from_json(const std::string& text, Graph* out);

// --- Binary (FlatBuffers-schema-aligned native codec; no flatc required) ----
[[nodiscard]] UAII_API Error graph_to_binary(const Graph& graph, std::string* out);
[[nodiscard]] UAII_API Error graph_from_binary(const std::string& bytes, Graph* out);

// --- Files ------------------------------------------------------------------
[[nodiscard]] UAII_API Error save_graph(const Graph& graph,
                                        const std::string& path,
                                        IrFormat format = IrFormat::Auto);

[[nodiscard]] UAII_API Error load_graph(const std::string& path,
                                        Graph* out,
                                        IrFormat format = IrFormat::Auto);

/// Round-trip helper used by tooling: load → save → load equality checks in tests.
[[nodiscard]] UAII_API Error round_trip_graph(const Graph& graph,
                                              IrFormat format,
                                              Graph* out);

// --- Text dumps for CLI -----------------------------------------------------
[[nodiscard]] UAII_API std::string graph_summary(const Graph& graph);
[[nodiscard]] UAII_API std::string graph_to_text(const Graph& graph);
[[nodiscard]] UAII_API std::string graph_to_dot(const Graph& graph);

[[nodiscard]] UAII_API std::string plan_to_json(const ExecutionPlan& plan);

}  // namespace ir
}  // namespace uaii
