#include "uaii/loaders/registry.hpp"

#include "uaii/loaders/gguf.hpp"
#include "uaii/loaders/safetensors.hpp"
#include "uaii/ir/serialize.hpp"

#include <cctype>
#include <cstring>
#include <fstream>

namespace uaii {
namespace loaders {
namespace {

std::string join_path(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
#if defined(_WIN32)
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  if (dir.back() == '/' || dir.back() == '\\') return dir + file;
  return dir + sep + file;
}

}  // namespace

void LoaderRegistry::register_loader(std::unique_ptr<IModelLoader> loader) {
  if (loader) {
    loaders_.push_back(std::move(loader));
  }
}

IModelLoader* LoaderRegistry::find_for_path(const std::string& path) const {
  for (const auto& l : loaders_) {
    if (l->accepts(path)) {
      return l.get();
    }
  }
  return nullptr;
}

std::vector<LoaderInfo> LoaderRegistry::list() const {
  std::vector<LoaderInfo> out;
  out.reserve(loaders_.size());
  for (const auto& l : loaders_) {
    out.push_back(l->info());
  }
  return out;
}

void LoaderRegistry::register_defaults() {
  loaders_.clear();
  register_loader(std::make_unique<GgufLoader>());
  register_loader(std::make_unique<SafetensorsLoader>());
}

LoaderRegistry& default_loaders() {
  static LoaderRegistry reg = [] {
    LoaderRegistry r;
    r.register_defaults();
    return r;
  }();
  return reg;
}

Error load_model(const std::string& path, ir::Graph* out) {
  auto* loader = default_loaders().find_for_path(path);
  if (loader == nullptr) {
    return Error::make(ErrorCode::NotFound,
                       "no loader accepts path: " + path);
  }
  return loader->load(path, out);
}

Error convert_model(const std::string& input_path, const std::string& output_path) {
  ir::Graph graph;
  Error err = load_model(input_path, &graph);
  if (!err.ok()) return err;
  return ir::save_graph(graph, output_path);
}

Error load_weight_ref_f32(const std::string& weight_ref,
                          const std::string& weights_dir,
                          const Shape& expected_shape,
                          float* dst,
                          std::size_t nbytes) {
  if (dst == nullptr || nbytes == 0) {
    return Error::make(ErrorCode::InvalidArgument, "weight load args");
  }

  const auto hash = weight_ref.find('#');
  if (hash != std::string::npos) {
    const std::string file = join_path(weights_dir, weight_ref.substr(0, hash));
    const std::string tensor = weight_ref.substr(hash + 1);
    const auto ext_pos = file.find_last_of('.');
    std::string ext = ext_pos == std::string::npos ? std::string{} : file.substr(ext_pos);
    for (char& c : ext) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    std::vector<float> values;
    Shape shape;
    Error err;
    if (ext == ".gguf") {
      GgufFile gf;
      err = gguf_read_header(file, &gf);
      if (!err.ok()) return err;
      err = gguf_load_tensor_f32(gf, tensor, &values, &shape);
    } else if (ext == ".safetensors") {
      SafetensorsFile sf;
      err = safetensors_read_header(file, &sf);
      if (!err.ok()) return err;
      err = safetensors_load_tensor_f32(sf, tensor, &values, &shape);
    } else {
      return Error::make(ErrorCode::InvalidArgument,
                         "unsupported weight container in ref: " + weight_ref);
    }
    if (!err.ok()) return err;

    if (!expected_shape.dims.empty() && expected_shape.dims != shape.dims) {
      // Allow if numel matches (transpose ambiguity).
      std::size_t e = 1, a = 1;
      for (auto d : expected_shape.dims) e *= static_cast<std::size_t>(d < 0 ? 0 : d);
      for (auto d : shape.dims) a *= static_cast<std::size_t>(d < 0 ? 0 : d);
      if (e != a) {
        return Error::make(ErrorCode::InvalidArgument,
                           "weight shape mismatch for " + weight_ref);
      }
    }
    if (values.size() * sizeof(float) != nbytes) {
      return Error::make(ErrorCode::InvalidArgument,
                         "weight byte size mismatch for " + weight_ref);
    }
    std::memcpy(dst, values.data(), nbytes);
    return Error::success();
  }

  // Raw binary blob
  const std::string path = join_path(weights_dir, weight_ref);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Error::make(ErrorCode::NotFound, "weight file not found: " + path);
  }
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(nbytes));
  if (static_cast<std::size_t>(in.gcount()) != nbytes) {
    return Error::make(ErrorCode::IoError, "short read for weight " + path);
  }
  return Error::success();
}

}  // namespace loaders
}  // namespace uaii
