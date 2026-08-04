#include "uaii/ir/serialize.hpp"

#include "uaii/ir/dtype.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace uaii {
namespace ir {

const char* to_string(IrFormat format) noexcept {
  switch (format) {
    case IrFormat::Json: return "json";
    case IrFormat::Binary: return "binary";
    default: return "auto";
  }
}

IrFormat infer_format_from_path(const std::string& path) noexcept {
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower.size() >= 10 && lower.compare(lower.size() - 10, 10, ".uaii.json") == 0) {
    return IrFormat::Json;
  }
  if (lower.size() >= 5 && lower.compare(lower.size() - 5, 5, ".json") == 0) {
    return IrFormat::Json;
  }
  if (lower.size() >= 5 && lower.compare(lower.size() - 5, 5, ".uaii") == 0) {
    return IrFormat::Binary;
  }
  return IrFormat::Auto;
}

Error save_graph(const Graph& graph, const std::string& path, IrFormat format) {
  IrFormat fmt = format;
  if (fmt == IrFormat::Auto) {
    fmt = infer_format_from_path(path);
    if (fmt == IrFormat::Auto) {
      fmt = IrFormat::Json;
    }
  }

  std::string payload;
  Error err = (fmt == IrFormat::Binary) ? graph_to_binary(graph, &payload)
                                        : graph_to_json(graph, &payload);
  if (!err.ok()) {
    return err;
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Error::make(ErrorCode::IoError, "failed to write " + path);
  }
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  if (!out) {
    return Error::make(ErrorCode::IoError, "failed while writing " + path);
  }
  return Error::ok();
}

Error load_graph(const std::string& path, Graph* out, IrFormat format) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out is null");
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Error::make(ErrorCode::NotFound, "IR file not found: " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string bytes = ss.str();

  IrFormat fmt = format;
  if (fmt == IrFormat::Auto) {
    fmt = infer_format_from_path(path);
    if (fmt == IrFormat::Auto) {
      if (bytes.size() >= 4 && bytes[0] == 'U' && bytes[1] == 'A' && bytes[2] == 'I' &&
          bytes[3] == 'R') {
        fmt = IrFormat::Binary;
      } else {
        fmt = IrFormat::Json;
      }
    }
  }

  if (fmt == IrFormat::Binary) {
    return graph_from_binary(bytes, out);
  }
  return graph_from_json(bytes, out);
}

Error round_trip_graph(const Graph& graph, IrFormat format, Graph* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out is null");
  }
  std::string payload;
  Error err = (format == IrFormat::Binary) ? graph_to_binary(graph, &payload)
                                           : graph_to_json(graph, &payload);
  if (!err.ok()) {
    return err;
  }
  if (format == IrFormat::Binary) {
    return graph_from_binary(payload, out);
  }
  return graph_from_json(payload, out);
}

std::string graph_summary(const Graph& graph) {
  std::ostringstream oss;
  oss << "UAII IR " << to_string(graph.version) << "\n"
      << "  name      : " << (graph.name.empty() ? "(unnamed)" : graph.name) << "\n"
      << "  producer  : " << graph.producer << "\n"
      << "  domain    : " << graph.domain << "\n"
      << "  tensors   : " << graph.tensors.size() << "\n"
      << "  nodes     : " << graph.nodes.size() << "\n"
      << "  inputs    : " << graph.inputs.size() << "\n"
      << "  outputs   : " << graph.outputs.size() << "\n"
      << "  metadata  : " << graph.metadata.size() << "\n";
  return oss.str();
}

std::string graph_to_text(const Graph& graph) {
  std::ostringstream oss;
  oss << graph_summary(graph) << "\n";

  oss << "Tensors:\n";
  for (const auto& t : graph.tensors) {
    oss << "  #" << t.id << " " << t.name << " " << uaii::to_string(t.dtype) << "[";
    for (std::size_t i = 0; i < t.shape.dims.size(); ++i) {
      if (i) oss << ",";
      oss << t.shape.dims[i];
    }
    oss << "]";
    if (t.is_weight) {
      oss << " weight";
      if (!t.weight_ref.empty()) {
        oss << "(" << t.weight_ref << ")";
      }
    }
    oss << "\n";
  }

  oss << "\nNodes:\n";
  for (const auto& n : graph.nodes) {
    oss << "  #" << n.id << " " << n.op_name << "@" << n.op_version;
    if (!n.name.empty()) {
      oss << " (" << n.name << ")";
    }
    oss << "\n    inputs : [";
    for (std::size_t i = 0; i < n.inputs.size(); ++i) {
      if (i) oss << ", ";
      oss << n.inputs[i];
    }
    oss << "]\n    outputs: [";
    for (std::size_t i = 0; i < n.outputs.size(); ++i) {
      if (i) oss << ", ";
      oss << n.outputs[i];
    }
    oss << "]\n";
    if (!n.attributes.empty()) {
      oss << "    attrs  :";
      for (const auto& a : n.attributes) {
        oss << " " << a.key;
      }
      oss << "\n";
    }
  }

  oss << "\nGraph I/O:\n  inputs : [";
  for (std::size_t i = 0; i < graph.inputs.size(); ++i) {
    if (i) oss << ", ";
    oss << graph.inputs[i];
  }
  oss << "]\n  outputs: [";
  for (std::size_t i = 0; i < graph.outputs.size(); ++i) {
    if (i) oss << ", ";
    oss << graph.outputs[i];
  }
  oss << "]\n";
  return oss.str();
}

std::string graph_to_dot(const Graph& graph) {
  std::ostringstream oss;
  oss << "digraph \"" << (graph.name.empty() ? "uaii_ir" : graph.name) << "\" {\n";
  oss << "  rankdir=LR;\n";
  oss << "  node [shape=box, fontname=\"Helvetica\"];\n";

  for (const auto& t : graph.tensors) {
    const char* color = t.is_weight ? "lightgoldenrod1" : "lightblue";
    oss << "  t" << t.id << " [label=\"T" << t.id << "\\n" << t.name << "\\n"
        << uaii::to_string(t.dtype) << "\", style=filled, fillcolor=" << color << "];\n";
  }
  for (const auto& n : graph.nodes) {
    oss << "  n" << n.id << " [label=\"" << n.op_name << "@" << n.op_version;
    if (!n.name.empty()) {
      oss << "\\n" << n.name;
    }
    oss << "\", shape=ellipse, style=filled, fillcolor=palegreen];\n";
    for (TensorId id : n.inputs) {
      oss << "  t" << id << " -> n" << n.id << ";\n";
    }
    for (TensorId id : n.outputs) {
      oss << "  n" << n.id << " -> t" << id << ";\n";
    }
  }
  oss << "}\n";
  return oss.str();
}

}  // namespace ir
}  // namespace uaii
