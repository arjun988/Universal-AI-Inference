#include "uaii/loaders/gguf.hpp"

#include "binary_io.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"

#include <cmath>
#include <cstring>

namespace uaii {
namespace loaders {
namespace {

enum class GgufValueType : std::uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

Error read_value(detail::ByteReader* r, GgufValueType type, GgufValue* out);

Error read_array(detail::ByteReader* r, GgufValue* out) {
  std::uint32_t et = 0;
  Error err = r->read_pod(&et);
  if (!err.ok()) return err;
  std::uint64_t n = 0;
  err = r->read_pod(&n);
  if (!err.ok()) return err;

  const auto at = static_cast<GgufValueType>(et);
  if (at == GgufValueType::String) {
    std::vector<std::string> vals;
    vals.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      std::string s;
      err = r->read_string(&s);
      if (!err.ok()) return err;
      vals.push_back(std::move(s));
    }
    *out = std::move(vals);
    return Error::ok();
  }
  if (at == GgufValueType::Int32) {
    std::vector<std::int32_t> vals(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      err = r->read_pod(&vals[static_cast<std::size_t>(i)]);
      if (!err.ok()) return err;
    }
    *out = std::move(vals);
    return Error::ok();
  }
  if (at == GgufValueType::Int64) {
    std::vector<std::int64_t> vals(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      err = r->read_pod(&vals[static_cast<std::size_t>(i)]);
      if (!err.ok()) return err;
    }
    *out = std::move(vals);
    return Error::ok();
  }
  if (at == GgufValueType::Float32) {
    std::vector<float> vals(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      err = r->read_pod(&vals[static_cast<std::size_t>(i)]);
      if (!err.ok()) return err;
    }
    *out = std::move(vals);
    return Error::ok();
  }

  // Skip unsupported array element types by reading scalars into discard.
  for (std::uint64_t i = 0; i < n; ++i) {
    GgufValue tmp;
    err = read_value(r, at, &tmp);
    if (!err.ok()) return err;
  }
  *out = std::monostate{};
  return Error::ok();
}

Error read_value(detail::ByteReader* r, GgufValueType type, GgufValue* out) {
  switch (type) {
    case GgufValueType::Uint8: {
      std::uint8_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Int8: {
      std::int8_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Uint16: {
      std::uint16_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Int16: {
      std::int16_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Uint32: {
      std::uint32_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Int32: {
      std::int32_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Float32: {
      float v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Bool: {
      std::int8_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = (v != 0);
      return Error::ok();
    }
    case GgufValueType::String: {
      std::string s;
      Error err = r->read_string(&s);
      if (!err.ok()) return err;
      *out = std::move(s);
      return Error::ok();
    }
    case GgufValueType::Array:
      return read_array(r, out);
    case GgufValueType::Uint64: {
      std::uint64_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Int64: {
      std::int64_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
    case GgufValueType::Float64: {
      double v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::ok();
    }
  }
  return Error::make(ErrorCode::InvalidArgument, "unknown GGUF value type");
}

void write_value(detail::ByteWriter* w, GgufValueType type, const GgufValue& value) {
  w->write_pod(static_cast<std::uint32_t>(type));
  switch (type) {
    case GgufValueType::Uint32:
      w->write_pod(std::get<std::uint32_t>(value));
      break;
    case GgufValueType::Int32:
      w->write_pod(std::get<std::int32_t>(value));
      break;
    case GgufValueType::Float32:
      w->write_pod(std::get<float>(value));
      break;
    case GgufValueType::Bool:
      w->write_pod(static_cast<std::int8_t>(std::get<bool>(value) ? 1 : 0));
      break;
    case GgufValueType::String:
      w->write_string(std::get<std::string>(value));
      break;
    case GgufValueType::Uint64:
      w->write_pod(std::get<std::uint64_t>(value));
      break;
    case GgufValueType::Int64:
      w->write_pod(std::get<std::int64_t>(value));
      break;
    default:
      break;
  }
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1u;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;
  std::uint32_t fbits = 0;
  if (exp == 0) {
    if (mant == 0) {
      fbits = sign << 31;
    } else {
      // subnormal
      float f = std::ldexp(static_cast<float>(mant), -24);
      if (sign) f = -f;
      return f;
    }
  } else if (exp == 31) {
    fbits = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    fbits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  }
  float out = 0;
  std::memcpy(&out, &fbits, sizeof(out));
  return out;
}

Shape dims_to_shape(const std::vector<std::uint64_t>& dims) {
  // GGUF stores dimensions in reverse order relative to typical row-major [rows, cols].
  Shape s;
  s.dims.reserve(dims.size());
  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    s.dims.push_back(static_cast<std::int64_t>(*it));
  }
  return s;
}

std::size_t numel_of(const std::vector<std::uint64_t>& dims) {
  std::size_t n = 1;
  for (auto d : dims) n *= static_cast<std::size_t>(d);
  return n;
}

}  // namespace

const char* to_string(GgufType type) noexcept {
  switch (type) {
    case GgufType::F32: return "f32";
    case GgufType::F16: return "f16";
    case GgufType::Q8_0: return "q8_0";
    case GgufType::I8: return "i8";
    case GgufType::I16: return "i16";
    case GgufType::I32: return "i32";
    default: return "other";
  }
}

DType gguf_to_dtype(GgufType type) noexcept {
  switch (type) {
    case GgufType::F32: return DType::F32;
    case GgufType::F16: return DType::F16;
    case GgufType::I8: return DType::I8;
    case GgufType::I32: return DType::I32;
    case GgufType::I64: return DType::I64;
    default: return DType::Unknown;
  }
}

std::size_t gguf_type_nbytes_per_elem(GgufType type) noexcept {
  switch (type) {
    case GgufType::F32: return 4;
    case GgufType::F16: return 2;
    case GgufType::I8: return 1;
    case GgufType::I16: return 2;
    case GgufType::I32: return 4;
    case GgufType::I64: return 8;
    case GgufType::F64: return 8;
    default: return 0;
  }
}

Error gguf_read_header(const std::string& path, GgufFile* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "gguf out null");
  }
  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(path, &r);
  if (!err.ok()) return err;

  char magic[4]{};
  err = r.read_bytes(magic, 4);
  if (!err.ok()) return err;
  if (std::strncmp(magic, "GGUF", 4) != 0) {
    return Error::make(ErrorCode::InvalidArgument, "not a GGUF file");
  }

  GgufFile file;
  file.path = path;
  err = r.read_pod(&file.version);
  if (!err.ok()) return err;

  std::uint64_t tensor_count = 0;
  std::uint64_t kv_count = 0;
  err = r.read_pod(&tensor_count);
  if (!err.ok()) return err;
  err = r.read_pod(&kv_count);
  if (!err.ok()) return err;

  for (std::uint64_t i = 0; i < kv_count; ++i) {
    std::string key;
    err = r.read_string(&key);
    if (!err.ok()) return err;
    std::uint32_t vt = 0;
    err = r.read_pod(&vt);
    if (!err.ok()) return err;
    GgufValue value;
    err = read_value(&r, static_cast<GgufValueType>(vt), &value);
    if (!err.ok()) return err;
    file.kv.emplace(std::move(key), std::move(value));
  }

  file.tensors.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo t;
    err = r.read_string(&t.name);
    if (!err.ok()) return err;
    std::uint32_t n_dims = 0;
    err = r.read_pod(&n_dims);
    if (!err.ok()) return err;
    t.dims.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      err = r.read_pod(&t.dims[d]);
      if (!err.ok()) return err;
    }
    std::uint32_t typ = 0;
    err = r.read_pod(&typ);
    if (!err.ok()) return err;
    t.type = static_cast<GgufType>(typ);
    err = r.read_pod(&t.offset);
    if (!err.ok()) return err;
    file.tensors.push_back(std::move(t));
  }

  // Align to 32 bytes for data section.
  std::size_t pos = r.tell();
  const std::size_t align = 32;
  pos = (pos + align - 1) & ~(align - 1);
  file.data_offset = pos;
  *out = std::move(file);
  return Error::ok();
}

Error gguf_load_tensor_f32(const GgufFile& file,
                           const std::string& tensor_name,
                           std::vector<float>* out,
                           Shape* out_shape) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "out null");
  }
  const GgufTensorInfo* info = nullptr;
  for (const auto& t : file.tensors) {
    if (t.name == tensor_name) {
      info = &t;
      break;
    }
  }
  if (info == nullptr) {
    return Error::make(ErrorCode::NotFound, "GGUF tensor not found: " + tensor_name);
  }

  const std::size_t n = numel_of(info->dims);
  const std::size_t elem = gguf_type_nbytes_per_elem(info->type);
  if (elem == 0 && info->type != GgufType::F32 && info->type != GgufType::F16) {
    return Error::make(ErrorCode::NotImplemented,
                       "GGUF type not supported for f32 load: " +
                           std::string(to_string(info->type)));
  }

  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(file.path, &r);
  if (!err.ok()) return err;
  r.seek(static_cast<std::size_t>(file.data_offset + info->offset));

  out->resize(n);
  if (info->type == GgufType::F32) {
    err = r.read_bytes(out->data(), n * 4);
    if (!err.ok()) return err;
  } else if (info->type == GgufType::F16) {
    std::vector<std::uint16_t> tmp(n);
    err = r.read_bytes(tmp.data(), n * 2);
    if (!err.ok()) return err;
    for (std::size_t i = 0; i < n; ++i) {
      (*out)[i] = f16_to_f32(tmp[i]);
    }
  } else {
    return Error::make(ErrorCode::NotImplemented, "unsupported GGUF dtype");
  }

  if (out_shape) {
    *out_shape = dims_to_shape(info->dims);
  }
  return Error::ok();
}

Error gguf_write_f32(
    const std::string& path,
    const std::unordered_map<std::string, GgufValue>& kv,
    const std::vector<std::pair<GgufTensorInfo, std::vector<float>>>& tensors) {
  detail::ByteWriter w;
  const char magic[4] = {'G', 'G', 'U', 'F'};
  w.write_pod(magic, 4);
  w.write_pod(std::uint32_t{3});  // version
  w.write_pod(static_cast<std::uint64_t>(tensors.size()));
  w.write_pod(static_cast<std::uint64_t>(kv.size()));

  for (const auto& kv_pair : kv) {
    w.write_string(kv_pair.first);
    if (std::holds_alternative<std::string>(kv_pair.second)) {
      write_value(&w, GgufValueType::String, kv_pair.second);
    } else if (std::holds_alternative<std::uint32_t>(kv_pair.second)) {
      write_value(&w, GgufValueType::Uint32, kv_pair.second);
    } else if (std::holds_alternative<std::int32_t>(kv_pair.second)) {
      write_value(&w, GgufValueType::Int32, kv_pair.second);
    } else if (std::holds_alternative<float>(kv_pair.second)) {
      write_value(&w, GgufValueType::Float32, kv_pair.second);
    } else if (std::holds_alternative<bool>(kv_pair.second)) {
      write_value(&w, GgufValueType::Bool, kv_pair.second);
    } else if (std::holds_alternative<std::uint64_t>(kv_pair.second)) {
      write_value(&w, GgufValueType::Uint64, kv_pair.second);
    } else {
      write_value(&w, GgufValueType::String, GgufValue{std::string{}});
    }
  }

  std::uint64_t offset = 0;
  for (const auto& tp : tensors) {
    GgufTensorInfo info = tp.first;
    info.type = GgufType::F32;
    info.offset = offset;
    w.write_string(info.name);
    w.write_pod(static_cast<std::uint32_t>(info.dims.size()));
    for (auto d : info.dims) w.write_pod(d);
    w.write_pod(static_cast<std::uint32_t>(info.type));
    w.write_pod(info.offset);
    offset += static_cast<std::uint64_t>(tp.second.size() * sizeof(float));
  }

  w.pad_to(32);
  for (const auto& tp : tensors) {
    if (!tp.second.empty()) {
      w.write_pod(tp.second.data(), tp.second.size() * sizeof(float));
    }
  }
  return w.save(path);
}

LoaderInfo GgufLoader::info() const {
  return LoaderInfo{"gguf", "1.0", {".gguf"}};
}

bool GgufLoader::accepts(const std::string& path) const {
  return detail::lower_ext(path) == ".gguf";
}

Error GgufLoader::load(const std::string& path, ir::Graph* out_graph) {
  if (out_graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out null");
  }
  GgufFile file;
  Error err = gguf_read_header(path, &file);
  if (!err.ok()) return err;

  const char* emb_names[] = {"token_embd.weight", "embed.weight",
                             "model.embed_tokens.weight"};
  const char* out_names[] = {"output.weight", "lm_head.weight"};
  const char* norm_names[] = {"output_norm.weight", "norm.weight"};

  auto find_name = [&](const char* const* names, std::size_t n) -> const GgufTensorInfo* {
    for (std::size_t i = 0; i < n; ++i) {
      for (const auto& t : file.tensors) {
        if (t.name == names[i]) return &t;
      }
    }
    return nullptr;
  };

  const GgufTensorInfo* emb = find_name(emb_names, 3);
  const GgufTensorInfo* lm = find_name(out_names, 2);
  const GgufTensorInfo* norm = find_name(norm_names, 2);

  ir::GraphBuilder b(emb && lm ? "gguf_tiny_lm" : "gguf_weights");
  b.set_producer("uaii-gguf-loader");
  b.set_metadata("source_format", "gguf");
  b.set_metadata("source_path", path);
  if (auto it = file.kv.find("general.name");
      it != file.kv.end() && std::holds_alternative<std::string>(it->second)) {
    b.set_metadata("model_name", std::get<std::string>(it->second));
  }

  if (emb && lm) {
    Shape emb_shape = dims_to_shape(emb->dims);
    Shape lm_shape = dims_to_shape(lm->dims);
    const std::int64_t dim = emb_shape.dims.size() > 1 ? emb_shape.dims[1] : 1;
    const std::int64_t vocab = lm_shape.dims.empty() ? emb_shape.dims[0] : lm_shape.dims[0];

    TensorId tokens = b.add_tensor("tokens", DType::F32, Shape{{1, 1}});
    TensorId emb_w =
        b.add_weight(emb->name, DType::F32, emb_shape, path + "#" + emb->name);
    TensorId hidden = b.add_tensor("hidden", DType::F32, Shape{{1, dim}});
    b.add_node("embed", "Embedding", "1", {tokens, emb_w}, {hidden});

    TensorId features = hidden;
    if (norm) {
      Shape nshape = dims_to_shape(norm->dims);
      TensorId nw =
          b.add_weight(norm->name, DType::F32, nshape, path + "#" + norm->name);
      TensorId nout = b.add_tensor("normed", DType::F32, Shape{{1, dim}});
      b.add_node("rms", "RMSNorm", "1", {hidden, nw}, {nout},
                 {ir::make_float_attr("eps", 1e-5)});
      features = nout;
    }

    TensorId lm_w =
        b.add_weight(lm->name, DType::F32, lm_shape, path + "#" + lm->name);
    TensorId logits = b.add_tensor("logits", DType::F32, Shape{{1, vocab}});
    b.add_node("lm_head", "MatMul", "1", {features, lm_w}, {logits},
               {ir::make_bool_attr("transpose_b", true)});
    TensorId probs = b.add_tensor("probs", DType::F32, Shape{{1, vocab}});
    b.add_node("softmax", "Softmax", "1", {logits}, {probs},
               {ir::make_int_attr("axis", -1)});
    b.set_inputs({tokens}).set_outputs({probs});
    b.set_metadata("architecture", "tiny_lm");
  } else {
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 1}});
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 1}});
    for (const auto& t : file.tensors) {
      Shape shape = dims_to_shape(t.dims);
      DType dt = gguf_to_dtype(t.type);
      if (dt == DType::Unknown) dt = DType::F32;
      (void)b.add_weight(t.name, dt == DType::F16 ? DType::F32 : dt, shape,
                         path + "#" + t.name);
    }
    b.add_node("id", "Identity", "1", {x}, {y});
    b.set_inputs({x}).set_outputs({y});
  }

  *out_graph = b.build();
  log::info("gguf") << "loaded " << file.tensors.size() << " tensors from " << path;
  return Error::ok();
}

}  // namespace loaders
}  // namespace uaii
