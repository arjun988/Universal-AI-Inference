#include "uaii/loaders/mlx.hpp"

#include "uaii/loaders/safetensors.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"

#include <cctype>
#include <fstream>
#include <filesystem>
#include <iterator>

namespace uaii {
namespace loaders {
namespace fs = std::filesystem;

LoaderInfo MlxLoader::info() const {
  return LoaderInfo{"mlx", "1.0", {".mlx"}};
}

bool MlxLoader::accepts(const std::string& path) const {
  // Directory with config.json + *.safetensors, or explicit .mlx marker file.
  std::error_code ec;
  if (fs::is_directory(path, ec)) {
    return fs::exists(fs::path(path) / "config.json", ec);
  }
  const auto pos = path.find_last_of('.');
  if (pos == std::string::npos) return false;
  std::string e = path.substr(pos);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e == ".mlx";
}

Error MlxLoader::load(const std::string& path, ir::Graph* out_graph) {
  if (out_graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out null");
  }
  fs::path root = path;
  if (fs::is_regular_file(root)) {
    root = root.parent_path();
  }
  fs::path weights;
  for (auto& p : fs::directory_iterator(root)) {
    if (p.path().extension() == ".safetensors") {
      weights = p.path();
      break;
    }
  }
  if (weights.empty()) {
    return Error::make(ErrorCode::NotFound, "no .safetensors under MLX root: " + path);
  }
  SafetensorsLoader st;
  Error err = st.load(weights.string(), out_graph);
  if (!err.ok()) return err;
  out_graph->metadata["source_format"] = "mlx";
  out_graph->metadata["source_path"] = path;
  const fs::path cfg = root / "config.json";
  if (fs::exists(cfg)) {
    std::ifstream in(cfg);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    out_graph->metadata["mlx_config"] = text.substr(0, 4096);
    // Pull num_attention_heads / num_hidden_layers if present as plain substrings.
    auto pull = [&](const char* key) {
      const std::string k = std::string("\"") + key + "\"";
      auto pos = text.find(k);
      if (pos == std::string::npos) return;
      pos = text.find(':', pos);
      if (pos == std::string::npos) return;
      std::size_t i = pos + 1;
      while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
      std::size_t j = i;
      while (j < text.size() && (std::isdigit(static_cast<unsigned char>(text[j])) || text[j] == '-'))
        ++j;
      if (j > i) out_graph->metadata[key] = text.substr(i, j - i);
    };
    pull("num_attention_heads");
    pull("num_hidden_layers");
    pull("hidden_size");
    pull("vocab_size");
  }
  log::info("mlx") << "imported safetensors " << weights.string();
  return Error::success();
}

}  // namespace loaders
}  // namespace uaii
