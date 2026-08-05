#include "uaii/loaders/onnx.hpp"

#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/serialize.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

namespace uaii {
namespace loaders {
namespace {

std::string lower_ext(const std::string& path) {
  const auto pos = path.find_last_of('.');
  if (pos == std::string::npos) return {};
  std::string e = path.substr(pos);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

/// Very small ONNX JSON (exported via onnx-simplifier / custom dump) or sidecar .uaii.json.
/// Binary protobuf ONNX is accepted by extension but requires structured parse of a
/// reduced op set from a companion `.uaii.onnx.json` when full protobuf deps are absent.
Error load_onnx_json_graph(const std::string& path, ir::Graph* out) {
  std::ifstream in(path);
  if (!in) return Error::make(ErrorCode::NotFound, "onnx json not found: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ir::graph_from_json(ss.str(), out);
}

}  // namespace

LoaderInfo OnnxLoader::info() const {
  return LoaderInfo{"onnx", "1.0", {".onnx", ".onnx.json"}};
}

bool OnnxLoader::accepts(const std::string& path) const {
  const auto e = lower_ext(path);
  return e == ".onnx" || path.size() > 10 && path.substr(path.size() - 10) == ".onnx.json";
}

Error OnnxLoader::load(const std::string& path, ir::Graph* out_graph) {
  if (out_graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out null");
  }
  // Prefer companion JSON IR dump for zero-dep builds.
  const std::string json_sidecar = path + ".uaii.json";
  {
    std::ifstream probe(json_sidecar);
    if (probe) {
      Error err = load_onnx_json_graph(json_sidecar, out_graph);
      if (err.ok()) {
        out_graph->metadata["source_format"] = "onnx";
        out_graph->metadata["source_path"] = path;
        return err;
      }
    }
  }
  if (lower_ext(path) == ".json" || path.find(".onnx.json") != std::string::npos) {
    Error err = load_onnx_json_graph(path, out_graph);
    if (err.ok()) {
      out_graph->metadata["source_format"] = "onnx";
      out_graph->metadata["source_path"] = path;
    }
    return err;
  }

  // Minimal binary ONNX: if file starts with protobuf and we find MatMul gemm patterns
  // we cannot fully parse without onnx proto — fail closed with guidance.
  std::ifstream in(path, std::ios::binary);
  if (!in) return Error::make(ErrorCode::NotFound, "onnx file not found: " + path);

#if defined(UAII_HAVE_ONNX)
  // Placeholder for full protobuf path when linked.
  return Error::make(ErrorCode::NotImplemented,
                     "UAII_HAVE_ONNX linked but full proto importer not compiled in this build");
#else
  return Error::make(
      ErrorCode::NotImplemented,
      "ONNX binary import requires companion '" + path +
          ".uaii.json' (export via torch.onnx + uaii convert helpers) or build with full ONNX proto");
#endif
}

}  // namespace loaders
}  // namespace uaii
