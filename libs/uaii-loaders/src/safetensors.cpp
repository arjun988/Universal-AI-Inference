#include "uaii/loaders/safetensors.hpp"

#include "binary_io.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"

#include <cctype>
#include <cstring>
#include <sstream>
#include <unordered_map>

namespace uaii {
namespace loaders {
namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

// Minimal JSON object parser for safetensors headers (objects/arrays/strings/numbers).
class MiniJson {
 public:
  explicit MiniJson(std::string text) : text_(std::move(text)) {}

  struct Value {
    enum Kind { Null, Number, String, Array, Object } kind = Null;
    double number = 0;
    std::string str;
    std::vector<Value> arr;
    std::unordered_map<std::string, Value> obj;
  };

  Error parse(Value* out) {
    skip();
    return parse_value(out);
  }

 private:
  std::string text_;
  std::size_t pos_ = 0;

  void skip() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  Error fail(const std::string& m) const {
    return Error::make(ErrorCode::InvalidArgument, "safetensors json: " + m);
  }

  Error parse_value(Value* out) {
    skip();
    if (pos_ >= text_.size()) return fail("eof");
    if (text_[pos_] == '{') return parse_object(out);
    if (text_[pos_] == '[') return parse_array(out);
    if (text_[pos_] == '"') return parse_string(out);
    if (text_[pos_] == '-' || std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      return parse_number(out);
    }
    if (text_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      *out = Value{};
      return Error::success();
    }
    return fail("bad token");
  }

  Error parse_string(Value* out) {
    if (text_[pos_] != '"') return fail("string");
    ++pos_;
    std::string s;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') {
        out->kind = Value::String;
        out->str = std::move(s);
        return Error::success();
      }
      if (c == '\\' && pos_ < text_.size()) {
        s.push_back(text_[pos_++]);
      } else {
        s.push_back(c);
      }
    }
    return fail("unterminated string");
  }

  Error parse_number(Value* out) {
    const std::size_t start = pos_;
    if (text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '.')) {
      ++pos_;
    }
    out->kind = Value::Number;
    out->number = std::stod(text_.substr(start, pos_ - start));
    return Error::success();
  }

  Error parse_array(Value* out) {
    if (text_[pos_] != '[') return fail("array");
    ++pos_;
    out->kind = Value::Array;
    skip();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return Error::success();
    }
    while (true) {
      Value item;
      Error err = parse_value(&item);
      if (!err.ok()) return err;
      out->arr.push_back(std::move(item));
      skip();
      if (pos_ < text_.size() && text_[pos_] == ']') {
        ++pos_;
        return Error::success();
      }
      if (pos_ >= text_.size() || text_[pos_] != ',') return fail("array comma");
      ++pos_;
    }
  }

  Error parse_object(Value* out) {
    if (text_[pos_] != '{') return fail("object");
    ++pos_;
    out->kind = Value::Object;
    skip();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return Error::success();
    }
    while (true) {
      Value key;
      Error err = parse_value(&key);
      if (!err.ok()) return err;
      if (key.kind != Value::String) return fail("key");
      skip();
      if (pos_ >= text_.size() || text_[pos_] != ':') return fail("colon");
      ++pos_;
      Value val;
      err = parse_value(&val);
      if (!err.ok()) return err;
      out->obj.emplace(std::move(key.str), std::move(val));
      skip();
      if (pos_ < text_.size() && text_[pos_] == '}') {
        ++pos_;
        return Error::success();
      }
      if (pos_ >= text_.size() || text_[pos_] != ',') return fail("object comma");
      ++pos_;
    }
  }
};

}  // namespace

DType safetensors_to_dtype(const std::string& dtype) noexcept {
  if (dtype == "F32" || dtype == "f32") return DType::F32;
  if (dtype == "F16" || dtype == "f16") return DType::F16;
  if (dtype == "BF16" || dtype == "bf16") return DType::BF16;
  if (dtype == "I8" || dtype == "i8") return DType::I8;
  if (dtype == "I32" || dtype == "i32") return DType::I32;
  if (dtype == "I64" || dtype == "i64") return DType::I64;
  if (dtype == "U8" || dtype == "u8") return DType::U8;
  if (dtype == "BOOL" || dtype == "bool") return DType::Bool;
  return DType::Unknown;
}

Error safetensors_read_header(const std::string& path, SafetensorsFile* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "safetensors out null");
  }
  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(path, &r);
  if (!err.ok()) return err;

  std::uint64_t header_len = 0;
  err = r.read_pod(&header_len);
  if (!err.ok()) return err;
  if (header_len == 0 || header_len > r.size()) {
    return Error::make(ErrorCode::InvalidArgument, "invalid safetensors header length");
  }
  std::string header(static_cast<std::size_t>(header_len), '\0');
  err = r.read_bytes(header.data(), static_cast<std::size_t>(header_len));
  if (!err.ok()) return err;

  MiniJson::Value root;
  MiniJson parser(header);
  err = parser.parse(&root);
  if (!err.ok()) return err;
  if (root.kind != MiniJson::Value::Object) {
    return Error::make(ErrorCode::InvalidArgument, "safetensors header not object");
  }

  SafetensorsFile file;
  file.path = path;
  file.header_size = header_len;
  file.data_offset = 8 + header_len;

  for (const auto& kv : root.obj) {
    if (kv.first == "__metadata__") {
      if (kv.second.kind == MiniJson::Value::Object) {
        for (const auto& m : kv.second.obj) {
          if (m.second.kind == MiniJson::Value::String) {
            file.metadata[m.first] = m.second.str;
          }
        }
      }
      continue;
    }
    if (kv.second.kind != MiniJson::Value::Object) continue;
    SafetensorsTensorInfo info;
    info.name = kv.first;
    auto dit = kv.second.obj.find("dtype");
    auto sit = kv.second.obj.find("shape");
    auto oit = kv.second.obj.find("data_offsets");
    if (dit == kv.second.obj.end() || sit == kv.second.obj.end() ||
        oit == kv.second.obj.end()) {
      return Error::make(ErrorCode::InvalidArgument,
                         "tensor entry incomplete: " + kv.first);
    }
    if (dit->second.kind != MiniJson::Value::String) {
      return Error::make(ErrorCode::InvalidArgument, "dtype not string");
    }
    info.dtype = dit->second.str;
    if (sit->second.kind != MiniJson::Value::Array) {
      return Error::make(ErrorCode::InvalidArgument, "shape not array");
    }
    for (const auto& d : sit->second.arr) {
      if (d.kind != MiniJson::Value::Number) {
        return Error::make(ErrorCode::InvalidArgument, "shape dim");
      }
      info.shape.push_back(static_cast<std::int64_t>(d.number));
    }
    if (oit->second.kind != MiniJson::Value::Array || oit->second.arr.size() != 2) {
      return Error::make(ErrorCode::InvalidArgument, "data_offsets");
    }
    info.data_offsets[0] = static_cast<std::uint64_t>(oit->second.arr[0].number);
    info.data_offsets[1] = static_cast<std::uint64_t>(oit->second.arr[1].number);
    file.tensors.push_back(std::move(info));
  }

  *out = std::move(file);
  return Error::success();
}

Error safetensors_load_tensor_f32(const SafetensorsFile& file,
                                  const std::string& tensor_name,
                                  std::vector<float>* out,
                                  Shape* out_shape) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "out null");
  }
  const SafetensorsTensorInfo* info = nullptr;
  for (const auto& t : file.tensors) {
    if (t.name == tensor_name) {
      info = &t;
      break;
    }
  }
  if (info == nullptr) {
    return Error::make(ErrorCode::NotFound, "safetensors tensor not found: " + tensor_name);
  }

  const std::uint64_t nbytes = info->data_offsets[1] - info->data_offsets[0];
  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(file.path, &r);
  if (!err.ok()) return err;
  r.seek(static_cast<std::size_t>(file.data_offset + info->data_offsets[0]));

  std::size_t numel = 1;
  for (auto d : info->shape) {
    if (d < 0) return Error::make(ErrorCode::InvalidArgument, "dynamic shape");
    numel *= static_cast<std::size_t>(d);
  }

  out->resize(numel);
  if (info->dtype == "F32" || info->dtype == "f32") {
    if (nbytes != numel * 4) {
      return Error::make(ErrorCode::InvalidArgument, "F32 size mismatch");
    }
    err = r.read_bytes(out->data(), static_cast<std::size_t>(nbytes));
    if (!err.ok()) return err;
  } else if (info->dtype == "F16" || info->dtype == "f16") {
    if (nbytes != numel * 2) {
      return Error::make(ErrorCode::InvalidArgument, "F16 size mismatch");
    }
    std::vector<std::uint16_t> tmp(numel);
    err = r.read_bytes(tmp.data(), static_cast<std::size_t>(nbytes));
    if (!err.ok()) return err;
    for (std::size_t i = 0; i < numel; ++i) {
      const std::uint16_t h = tmp[i];
      const std::uint32_t sign = (h >> 15) & 1u;
      const std::uint32_t exp = (h >> 10) & 0x1Fu;
      const std::uint32_t mant = h & 0x3FFu;
      std::uint32_t fbits = 0;
      if (exp == 0) {
        fbits = sign << 31;
      } else if (exp == 31) {
        fbits = (sign << 31) | 0x7F800000u | (mant << 13);
      } else {
        fbits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
      }
      std::memcpy(&(*out)[i], &fbits, sizeof(float));
    }
  } else {
    return Error::make(ErrorCode::NotImplemented,
                       "safetensors dtype not supported for f32 load: " + info->dtype);
  }

  if (out_shape) {
    out_shape->dims = info->shape;
  }
  return Error::success();
}

Error safetensors_write_f32(
    const std::string& path,
    const std::unordered_map<std::string, std::pair<Shape, std::vector<float>>>& tensors,
    const std::unordered_map<std::string, std::string>& metadata) {
  // Build JSON header
  std::ostringstream json;
  json << "{";
  bool first = true;
  if (!metadata.empty()) {
    json << "\"__metadata__\":{";
    bool mf = true;
    for (const auto& m : metadata) {
      if (!mf) json << ",";
      mf = false;
      json << "\"" << json_escape(m.first) << "\":\"" << json_escape(m.second) << "\"";
    }
    json << "}";
    first = false;
  }

  std::uint64_t offset = 0;
  struct Planned {
    std::string name;
    Shape shape;
    std::vector<float> data;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  std::vector<Planned> planned;
  planned.reserve(tensors.size());
  for (const auto& kv : tensors) {
    Planned p;
    p.name = kv.first;
    p.shape = kv.second.first;
    p.data = kv.second.second;
    p.begin = offset;
    p.end = offset + static_cast<std::uint64_t>(p.data.size() * sizeof(float));
    offset = p.end;
    planned.push_back(std::move(p));
  }

  for (const auto& p : planned) {
    if (!first) json << ",";
    first = false;
    json << "\"" << json_escape(p.name) << "\":{";
    json << "\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t i = 0; i < p.shape.dims.size(); ++i) {
      if (i) json << ",";
      json << p.shape.dims[i];
    }
    json << "],\"data_offsets\":[" << p.begin << "," << p.end << "]}";
  }
  json << "}";
  const std::string header = json.str();

  detail::ByteWriter w;
  w.write_pod(static_cast<std::uint64_t>(header.size()));
  w.write_pod(header.data(), header.size());
  for (const auto& p : planned) {
    if (!p.data.empty()) {
      w.write_pod(p.data.data(), p.data.size() * sizeof(float));
    }
  }
  return w.save(path);
}

LoaderInfo SafetensorsLoader::info() const {
  return LoaderInfo{"safetensors", "1.0", {".safetensors"}};
}

bool SafetensorsLoader::accepts(const std::string& path) const {
  return detail::lower_ext(path) == ".safetensors";
}

Error SafetensorsLoader::load(const std::string& path, ir::Graph* out_graph) {
  if (out_graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out null");
  }
  SafetensorsFile file;
  Error err = safetensors_read_header(path, &file);
  if (!err.ok()) return err;

  const SafetensorsTensorInfo* emb = nullptr;
  const SafetensorsTensorInfo* lm = nullptr;
  const SafetensorsTensorInfo* norm = nullptr;
  for (const auto& t : file.tensors) {
    if (t.name == "embed.weight" || t.name == "token_embd.weight" ||
        t.name == "model.embed_tokens.weight") {
      emb = &t;
    }
    if (t.name == "lm_head.weight" || t.name == "output.weight") {
      lm = &t;
    }
    if (t.name == "norm.weight" || t.name == "output_norm.weight") {
      norm = &t;
    }
  }

  auto find_st = [&](const std::string& name) -> const SafetensorsTensorInfo* {
    for (const auto& t : file.tensors) {
      if (t.name == name) return &t;
    }
    return nullptr;
  };
  const bool has_blk0 = find_st("blk.0.attn_q.weight") != nullptr ||
                        find_st("model.layers.0.self_attn.q_proj.weight") != nullptr;

  ir::GraphBuilder b(emb && lm ? (has_blk0 ? "safetensors_transformer" : "safetensors_tiny_lm")
                               : "safetensors_weights");
  b.set_producer("uaii-safetensors-loader");
  b.set_metadata("source_format", "safetensors");
  b.set_metadata("source_path", path);
  for (const auto& m : file.metadata) {
    b.set_metadata(m.first, m.second);
  }

  if (emb && lm) {
    const std::int64_t dim = emb->shape.size() > 1 ? emb->shape[1] : 1;
    const std::int64_t vocab = lm->shape.empty() ? emb->shape[0] : lm->shape[0];
    std::int64_t n_heads = 1;
    if (auto it = file.metadata.find("num_attention_heads"); it != file.metadata.end()) {
      try {
        n_heads = std::stoll(it->second);
      } catch (...) {
      }
    }
    std::int64_t n_layers = 0;
    for (std::int64_t i = 0; i < 64; ++i) {
      if (find_st("blk." + std::to_string(i) + ".attn_q.weight") ||
          find_st("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight")) {
        n_layers = i + 1;
      } else {
        break;
      }
    }

    TensorId tokens = b.add_tensor("tokens", DType::F32, Shape{{1, 1}});
    TensorId emb_w =
        b.add_weight(emb->name, DType::F32, Shape{emb->shape}, path + "#" + emb->name);
    TensorId hidden = b.add_tensor("hidden", DType::F32, Shape{{1, dim}});
    b.add_node("embed", "Embedding", "1", {tokens, emb_w}, {hidden});

    TensorId features = hidden;
    auto add_w = [&](const std::string& name) -> TensorId {
      const auto* ti = find_st(name);
      if (!ti) return 0;
      return b.add_weight(name, DType::F32, Shape{ti->shape}, path + "#" + name);
    };

    for (std::int64_t li = 0; li < n_layers; ++li) {
      const std::string blk = "blk." + std::to_string(li) + ".";
      const std::string hf = "model.layers." + std::to_string(li) + ".";
      TensorId attn_norm = add_w(blk + "attn_norm.weight");
      if (!attn_norm) attn_norm = add_w(hf + "input_layernorm.weight");
      TensorId wq = add_w(blk + "attn_q.weight");
      if (!wq) wq = add_w(hf + "self_attn.q_proj.weight");
      TensorId wk = add_w(blk + "attn_k.weight");
      if (!wk) wk = add_w(hf + "self_attn.k_proj.weight");
      TensorId wv = add_w(blk + "attn_v.weight");
      if (!wv) wv = add_w(hf + "self_attn.v_proj.weight");
      TensorId wo = add_w(blk + "attn_output.weight");
      if (!wo) wo = add_w(hf + "self_attn.o_proj.weight");
      TensorId ffn_norm = add_w(blk + "ffn_norm.weight");
      if (!ffn_norm) ffn_norm = add_w(hf + "post_attention_layernorm.weight");
      TensorId w_gate = add_w(blk + "ffn_gate.weight");
      if (!w_gate) w_gate = add_w(hf + "mlp.gate_proj.weight");
      TensorId w_up = add_w(blk + "ffn_up.weight");
      if (!w_up) w_up = add_w(hf + "mlp.up_proj.weight");
      TensorId w_down = add_w(blk + "ffn_down.weight");
      if (!w_down) w_down = add_w(hf + "mlp.down_proj.weight");
      if (!attn_norm || !wq || !wk || !wv || !wo) break;

      const std::string pfx = "l" + std::to_string(li) + ".";
      TensorId n1 = b.add_tensor(pfx + "n1", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "rms1", "RMSNorm", "1", {features, attn_norm}, {n1},
                 {ir::make_float_attr("eps", 1e-5)});
      TensorId q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, dim}});
      TensorId k = b.add_tensor(pfx + "k", DType::F32, Shape{{1, dim}});
      TensorId v = b.add_tensor(pfx + "v", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "q_proj", "MatMul", "1", {n1, wq}, {q},
                 {ir::make_bool_attr("transpose_b", true)});
      b.add_node(pfx + "k_proj", "MatMul", "1", {n1, wk}, {k},
                 {ir::make_bool_attr("transpose_b", true)});
      b.add_node(pfx + "v_proj", "MatMul", "1", {n1, wv}, {v},
                 {ir::make_bool_attr("transpose_b", true)});
      TensorId attn = b.add_tensor(pfx + "attn", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "attn", "Attention", "1", {q, k, v}, {attn},
                 {ir::make_int_attr("num_heads", n_heads),
                  ir::make_bool_attr("causal", true),
                  ir::make_bool_attr("use_kv_cache", true),
                  ir::make_int_attr("layer_id", li)});
      TensorId attn_o = b.add_tensor(pfx + "attn_o", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "o_proj", "MatMul", "1", {attn, wo}, {attn_o},
                 {ir::make_bool_attr("transpose_b", true)});
      TensorId resid1 = b.add_tensor(pfx + "resid1", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "add1", "Add", "1", {features, attn_o}, {resid1});
      if (w_gate && w_up && w_down) {
        TensorId mlp_in = resid1;
        if (ffn_norm) {
          TensorId n2 = b.add_tensor(pfx + "n2", DType::F32, Shape{{1, dim}});
          b.add_node(pfx + "rms2", "RMSNorm", "1", {resid1, ffn_norm}, {n2},
                     {ir::make_float_attr("eps", 1e-5)});
          mlp_in = n2;
        }
        const auto* up_info = find_st(blk + "ffn_up.weight");
        if (!up_info) up_info = find_st(hf + "mlp.up_proj.weight");
        const std::int64_t inter =
            up_info && up_info->shape.size() == 2
                ? (up_info->shape[0] == dim ? up_info->shape[1] : up_info->shape[0])
                : dim * 4;
        TensorId gate = b.add_tensor(pfx + "gate", DType::F32, Shape{{1, inter}});
        TensorId up = b.add_tensor(pfx + "up", DType::F32, Shape{{1, inter}});
        b.add_node(pfx + "ffn_gate", "MatMul", "1", {mlp_in, w_gate}, {gate},
                   {ir::make_bool_attr("transpose_b", true)});
        b.add_node(pfx + "ffn_up", "MatMul", "1", {mlp_in, w_up}, {up},
                   {ir::make_bool_attr("transpose_b", true)});
        TensorId gate_act = b.add_tensor(pfx + "gate_act", DType::F32, Shape{{1, inter}});
        b.add_node(pfx + "silu", "Silu", "1", {gate}, {gate_act});
        TensorId ff_hid = b.add_tensor(pfx + "ff_hid", DType::F32, Shape{{1, inter}});
        b.add_node(pfx + "ff_mul", "Mul", "1", {gate_act, up}, {ff_hid});
        TensorId ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "ffn_down", "MatMul", "1", {ff_hid, w_down}, {ff_out},
                   {ir::make_bool_attr("transpose_b", true)});
        TensorId resid2 = b.add_tensor(pfx + "resid2", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "add2", "Add", "1", {resid1, ff_out}, {resid2});
        features = resid2;
      } else {
        features = resid1;
      }
    }

    if (norm) {
      TensorId nw =
          b.add_weight(norm->name, DType::F32, Shape{norm->shape}, path + "#" + norm->name);
      TensorId nout = b.add_tensor("normed", DType::F32, Shape{{1, dim}});
      b.add_node("rms_out", "RMSNorm", "1", {features, nw}, {nout},
                 {ir::make_float_attr("eps", 1e-5)});
      features = nout;
    }

    TensorId lm_w =
        b.add_weight(lm->name, DType::F32, Shape{lm->shape}, path + "#" + lm->name);
    TensorId logits = b.add_tensor("logits", DType::F32, Shape{{1, vocab}});
    b.add_node("lm_head", "MatMul", "1", {features, lm_w}, {logits},
               {ir::make_bool_attr("transpose_b", true)});
    TensorId probs = b.add_tensor("probs", DType::F32, Shape{{1, vocab}});
    b.add_node("softmax", "Softmax", "1", {logits}, {probs},
               {ir::make_int_attr("axis", -1)});
    b.set_inputs({tokens}).set_outputs({probs});
    b.set_metadata("architecture", n_layers > 0 ? "transformer" : "tiny_lm");
    b.set_metadata("n_layers", std::to_string(n_layers));
    b.set_metadata("n_heads", std::to_string(n_heads));
    b.set_metadata("embedding_length", std::to_string(dim));
    if (auto it = file.metadata.find("max_position_embeddings");
        it != file.metadata.end()) {
      b.set_metadata("max_context", it->second);
      b.set_metadata("context_length", it->second);
    }
  } else {
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 1}});
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 1}});
    for (const auto& t : file.tensors) {
      DType dt = safetensors_to_dtype(t.dtype);
      if (dt == DType::F16 || dt == DType::Unknown) dt = DType::F32;
      (void)b.add_weight(t.name, dt, Shape{t.shape}, path + "#" + t.name);
    }
    b.add_node("id", "Identity", "1", {x}, {y});
    b.set_inputs({x}).set_outputs({y});
  }

  *out_graph = b.build();
  log::info("safetensors") << "loaded " << file.tensors.size() << " tensors from " << path;
  return Error::success();
}

}  // namespace loaders
}  // namespace uaii
