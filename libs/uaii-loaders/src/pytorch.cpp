#include "uaii/loaders/pytorch.hpp"

#include "uaii/loaders/onnx.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/serialize.hpp"

#include <fstream>
#include <sstream>

namespace uaii {
namespace loaders {

LoaderInfo PytorchLoader::info() const {
  return LoaderInfo{"pytorch", "1.0", {".pt", ".pth"}};
}

bool PytorchLoader::accepts(const std::string& path) const {
  const auto pos = path.find_last_of('.');
  if (pos == std::string::npos) return false;
  std::string e = path.substr(pos);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e == ".pt" || e == ".pth";
}

Error PytorchLoader::load(const std::string& path, ir::Graph* out_graph) {
  if (out_graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out null");
  }
  // Documented path: torch.onnx.export → model.onnx (+ optional .uaii.json)
  const std::string onnx_path = path + ".onnx";
  {
    std::ifstream probe(onnx_path);
    if (probe) {
      OnnxLoader onnx;
      Error err = onnx.load(onnx_path, out_graph);
      if (err.ok()) {
        out_graph->metadata["source_format"] = "pytorch";
        out_graph->metadata["source_path"] = path;
      }
      return err;
    }
  }
  const std::string json_path = path + ".uaii.json";
  {
    std::ifstream probe(json_path);
    if (probe) {
      std::ostringstream ss;
      ss << probe.rdbuf();
      Error err = ir::graph_from_json(ss.str(), out_graph);
      if (err.ok()) {
        out_graph->metadata["source_format"] = "pytorch";
        out_graph->metadata["source_path"] = path;
      }
      return err;
    }
  }

#if defined(UAII_HAVE_LIBTORCH)
  return Error::make(ErrorCode::NotImplemented,
                     "LibTorch linked but TorchScript importer not enabled in this build");
#else
  return Error::make(ErrorCode::NotImplemented,
                     "PyTorch .pt requires exported '" + path +
                         ".onnx' or '" + path +
                         ".uaii.json' (torch.onnx.export / UAII_WITH_LIBTORCH)");
#endif
}

}  // namespace loaders
}  // namespace uaii
