#include "uaii/loaders/hf.hpp"

#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/json.hpp"
#include "uaii/loaders/safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace uaii {
namespace loaders {
namespace {

namespace fs = std::filesystem;
namespace json = uaii::ir::json;

constexpr const char* kSupportedArchs[] = {
    "LlamaForCausalLM",   "MistralForCausalLM", "Qwen2ForCausalLM",  "Qwen3ForCausalLM",
    "GemmaForCausalLM",   "Gemma2ForCausalLM",  "Gemma3ForCausalLM", "Phi3ForCausalLM",
    "PhiForCausalLM",     "InternLM2ForCausalLM", "InternLMForCausalLM", "StableLmForCausalLM",
    "CohereForCausalLM",  "OlmoForCausalLM",    "Olmo2ForCausalLM",  "GraniteForCausalLM",
    "ExaoneForCausalLM",
};

std::string lower_ascii(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string join_path(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
  if (file.empty()) return dir;
  const fs::path p(file);
  if (p.is_absolute()) return file;
#if defined(_WIN32)
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  if (dir.back() == '/' || dir.back() == '\\') return dir + file;
  return dir + sep + file;
}

std::string read_text_file(const std::string& path, Error* err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *err = Error::make(ErrorCode::NotFound, "cannot open file: " + path);
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *err = Error::success();
  return ss.str();
}

const json::Value* find_key(const json::Object& obj, std::initializer_list<const char*> keys) {
  for (const char* k : keys) {
    if (const json::Value* v = json::get(obj, k)) return v;
  }
  return nullptr;
}

Error require_i64(const json::Object& obj,
                  std::initializer_list<const char*> keys,
                  std::int64_t* out,
                  const char* field_label) {
  const json::Value* v = find_key(obj, keys);
  if (v == nullptr || !v->is_number()) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string("config.json missing required field: ") + field_label);
  }
  *out = static_cast<std::int64_t>(v->number);
  if (*out <= 0) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string("config.json field must be > 0: ") + field_label);
  }
  return Error::success();
}

Error require_f32(const json::Object& obj,
                  std::initializer_list<const char*> keys,
                  float* out,
                  const char* field_label) {
  const json::Value* v = find_key(obj, keys);
  if (v == nullptr || !v->is_number()) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string("config.json missing required field: ") + field_label);
  }
  *out = static_cast<float>(v->number);
  if (!(*out > 0.f)) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string("config.json field must be > 0: ") + field_label);
  }
  return Error::success();
}

bool has_reject_key(const json::Object& obj, const char* key) {
  const json::Value* v = json::get(obj, key);
  if (v == nullptr || v->is_null()) return false;
  if (v->is_number() && v->number == 0.0) return false;
  return true;
}

int rope_mode_for_arch(const std::string& arch) {
  const std::string a = lower_ascii(arch);
  if (a.find("qwen") != std::string::npos || a.find("gemma") != std::string::npos ||
      a.find("phi") != std::string::npos || a.find("olmo") != std::string::npos ||
      a.find("stablelm") != std::string::npos) {
    return 2;
  }
  return 0;
}

std::string detect_chat_template_family(const std::string& chat_template,
                                        const std::string& arch) {
  if (!chat_template.empty()) {
    if (chat_template.find("<|start_header_id|>") != std::string::npos) return "llama3";
    if (chat_template.find("<start_of_turn>") != std::string::npos) return "gemma";
    if (chat_template.find("[INST]") != std::string::npos) return "mistral";
    if (chat_template.find("<|im_start|>") != std::string::npos) return "chatml";
  }
  const std::string a = lower_ascii(arch);
  if (a.find("gemma") != std::string::npos) return "gemma";
  if (a.find("mistral") != std::string::npos) return "mistral";
  if (a.find("llama") != std::string::npos) return "llama3";
  return "chatml";
}

std::int64_t apply_layer_cap(std::int64_t model_layers) {
  constexpr std::int64_t kAbsoluteMax = 512;
  std::int64_t n = model_layers;
  if (n > kAbsoluteMax) {
    log::warn("hf") << "layer count " << n << " exceeds absolute max " << kAbsoluteMax
                    << ", clamping";
    n = kAbsoluteMax;
  }
  const char* env = std::getenv("UAII_MAX_LAYERS");
  if (env != nullptr && env[0] != '\0') {
    const int cap = std::atoi(env);
    if (cap > 0 && n > cap) {
      log::warn("hf") << "UAII_MAX_LAYERS=" << cap << " clamping layers from " << n;
      n = cap;
    }
  }
  return n;
}

std::string load_optional_chat_template(const std::string& dir) {
  const std::string path = join_path(dir, "tokenizer_config.json");
  std::error_code ec;
  if (!fs::exists(path, ec)) return {};
  Error err;
  const std::string text = read_text_file(path, &err);
  if (!err.ok()) return {};
  json::Value root;
  if (!json::parse(text, &root).ok() || !root.is_object()) return {};
  const json::Value* ct = json::get(root.obj, "chat_template");
  if (ct && ct->is_string()) return ct->str;
  return {};
}

struct ShardCache {
  std::unordered_map<std::string, SafetensorsFile> files;

  Error ensure(const std::string& abs_path) {
    if (files.count(abs_path)) return Error::success();
    SafetensorsFile sf;
    Error err = safetensors_read_header(abs_path, &sf);
    if (!err.ok()) return err;
    files.emplace(abs_path, std::move(sf));
    return Error::success();
  }

  const SafetensorsTensorInfo* find(const std::string& abs_path,
                                    const std::string& tensor) {
    auto it = files.find(abs_path);
    if (it == files.end()) return nullptr;
    for (const auto& t : it->second.tensors) {
      if (t.name == tensor) return &t;
    }
    return nullptr;
  }
};

}  // namespace

bool hf_architecture_supported(const std::string& arch_name) noexcept {
  for (const char* a : kSupportedArchs) {
    if (arch_name == a) return true;
  }
  return false;
}

Error hf_parse_config_json(const std::string& text, HfConfig* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "HfConfig out null");
  }
  *out = HfConfig{};

  json::Value root_v;
  Error err = json::parse(text, &root_v);
  if (!err.ok()) return err;

  const json::Object* root = nullptr;
  err = json::require_object(root_v, &root, "config.json");
  if (!err.ok()) return err;

  static const char* kReject[] = {
      "num_local_experts", "num_experts", "moe_intermediate_size", "kv_lora_rank", "q_lora_rank",
  };
  for (const char* k : kReject) {
    if (has_reject_key(*root, k)) {
      if (std::string(k) == "kv_lora_rank" || std::string(k) == "q_lora_rank") {
        return Error::make(
            ErrorCode::NotImplemented,
            std::string("config.json indicates MLA (") + k +
                "); HF MLA is not built yet — use DeepSeek GGUF with UAII MLA expand path");
      }
      return Error::make(ErrorCode::NotImplemented,
                         std::string("config.json indicates unsupported MoE/MLA field: ") + k);
    }
  }

  const json::Value* arches = json::get(*root, "architectures");
  if (arches == nullptr || !arches->is_array() || arches->arr.empty() ||
      !arches->arr[0].is_string() || arches->arr[0].str.empty()) {
    return Error::make(ErrorCode::InvalidArgument,
                       "config.json missing required field: architectures[0]");
  }
  out->architecture = arches->arr[0].str;
  if (!hf_architecture_supported(out->architecture)) {
    return Error::make(ErrorCode::NotImplemented,
                       "unsupported Hugging Face architecture: " + out->architecture);
  }

  if (const json::Value* mt = json::get(*root, "model_type"); mt && mt->is_string()) {
    out->model_type = mt->str;
  }

  err = require_i64(*root, {"hidden_size", "n_embd", "d_model"}, &out->hidden_size, "hidden_size");
  if (!err.ok()) return err;
  err = require_i64(*root, {"num_hidden_layers", "n_layer", "n_layers"}, &out->num_hidden_layers,
                    "num_hidden_layers");
  if (!err.ok()) return err;
  err = require_i64(*root, {"num_attention_heads", "n_head"}, &out->num_attention_heads,
                    "num_attention_heads");
  if (!err.ok()) return err;
  err = require_i64(*root, {"intermediate_size", "n_inner", "ffn_dim"}, &out->intermediate_size,
                    "intermediate_size");
  if (!err.ok()) return err;
  err = require_i64(*root, {"vocab_size"}, &out->vocab_size, "vocab_size");
  if (!err.ok()) return err;
  err = require_i64(*root, {"max_position_embeddings", "n_positions", "max_sequence_length"},
                    &out->max_position_embeddings, "max_position_embeddings");
  if (!err.ok()) return err;
  err = require_f32(*root, {"rms_norm_eps", "rms_norm_epsilon", "layer_norm_eps", "layer_norm_epsilon"},
                    &out->rms_norm_eps, "rms_norm_eps");
  if (!err.ok()) return err;

  if (const json::Value* kv = find_key(*root, {"num_key_value_heads"})) {
    if (!kv->is_number() || kv->number <= 0) {
      return Error::make(ErrorCode::InvalidArgument, "invalid num_key_value_heads");
    }
    out->num_key_value_heads = static_cast<std::int64_t>(kv->number);
  } else {
    out->num_key_value_heads = out->num_attention_heads;
  }

  if (const json::Value* th = find_key(*root, {"rope_theta"})) {
    if (!th->is_number() || !(th->number > 0.0)) {
      return Error::make(ErrorCode::InvalidArgument, "invalid rope_theta");
    }
    out->rope_theta = static_cast<float>(th->number);
  } else {
    out->rope_theta = 10000.f;
  }

  if (const json::Value* hd = find_key(*root, {"head_dim"})) {
    if (!hd->is_number() || hd->number <= 0) {
      return Error::make(ErrorCode::InvalidArgument, "invalid head_dim");
    }
    out->head_dim = static_cast<std::int64_t>(hd->number);
  } else {
    out->head_dim = out->hidden_size / out->num_attention_heads;
    if (out->head_dim <= 0) {
      return Error::make(ErrorCode::InvalidArgument, "cannot derive head_dim from hidden/heads");
    }
  }

  if (const json::Value* tie = json::get(*root, "tie_word_embeddings");
      tie && tie->is_bool()) {
    out->tie_word_embeddings = tie->b;
  }

  out->rope_mode = rope_mode_for_arch(out->architecture);
  return Error::success();
}

Error hf_load_config_file(const std::string& path, HfConfig* out) {
  Error err;
  const std::string text = read_text_file(path, &err);
  if (!err.ok()) return err;
  return hf_parse_config_json(text, out);
}

Error hf_load_weight_index(const std::string& dir,
                           std::unordered_map<std::string, std::string>* weight_map,
                           std::vector<std::string>* shard_files) {
  if (weight_map == nullptr || shard_files == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "weight index out null");
  }
  weight_map->clear();
  shard_files->clear();

  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    return Error::make(ErrorCode::InvalidArgument, "not a directory: " + dir);
  }

  fs::path index_path;
  const fs::path preferred = fs::path(dir) / "model.safetensors.index.json";
  if (fs::exists(preferred, ec)) {
    index_path = preferred;
  } else {
    for (const auto& ent : fs::directory_iterator(dir, ec)) {
      if (!ent.is_regular_file(ec)) continue;
      const std::string name = ent.path().filename().string();
      if (name.size() >= 20 &&
          name.compare(name.size() - 20, 20, ".safetensors.index.json") == 0) {
        index_path = ent.path();
        break;
      }
    }
  }

  if (!index_path.empty()) {
    Error err;
    const std::string text = read_text_file(index_path.string(), &err);
    if (!err.ok()) return err;
    json::Value root_v;
    err = json::parse(text, &root_v);
    if (!err.ok()) return err;
    const json::Object* root = nullptr;
    err = json::require_object(root_v, &root, "weight index");
    if (!err.ok()) return err;
    const json::Value* wm = json::get(*root, "weight_map");
    if (wm == nullptr || !wm->is_object()) {
      return Error::make(ErrorCode::InvalidArgument,
                         "weight index missing weight_map: " + index_path.string());
    }
    std::unordered_map<std::string, bool> seen_shards;
    for (const auto& kv : wm->obj) {
      if (!kv.second.is_string() || kv.second.str.empty()) {
        return Error::make(ErrorCode::InvalidArgument,
                           "invalid weight_map entry for tensor: " + kv.first);
      }
      (*weight_map)[kv.first] = kv.second.str;
      if (!seen_shards[kv.second.str]) {
        seen_shards[kv.second.str] = true;
        shard_files->push_back(kv.second.str);
      }
    }
    if (weight_map->empty()) {
      return Error::make(ErrorCode::InvalidArgument, "empty weight_map in " + index_path.string());
    }
    return Error::success();
  }

  // Fallback: scan *.safetensors and build tensor → shard map from headers.
  std::vector<fs::path> shards;
  for (const auto& ent : fs::directory_iterator(dir, ec)) {
    if (!ent.is_regular_file(ec)) continue;
    if (ent.path().extension() == ".safetensors") {
      shards.push_back(ent.path());
    }
  }
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) {
    return Error::make(ErrorCode::NotFound, "no .safetensors or index.json under: " + dir);
  }

  for (const auto& sp : shards) {
    const std::string rel = sp.filename().string();
    shard_files->push_back(rel);
    SafetensorsFile sf;
    Error err = safetensors_read_header(sp.string(), &sf);
    if (!err.ok()) return err;
    for (const auto& t : sf.tensors) {
      (*weight_map)[t.name] = rel;
    }
  }
  if (weight_map->empty()) {
    return Error::make(ErrorCode::InvalidArgument, "no tensors found in safetensors under: " + dir);
  }
  return Error::success();
}

LoaderInfo HfLoader::info() const {
  return LoaderInfo{"huggingface", "1.0", {}};
}

bool HfLoader::accepts(const std::string& path) const {
  std::error_code ec;
  if (!fs::is_directory(path, ec)) return false;
  return fs::exists(fs::path(path) / "config.json", ec);
}

Error HfLoader::load(const std::string& path, ir::Graph* out_graph) {
  if (out_graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out null");
  }
  std::error_code ec;
  if (!fs::is_directory(path, ec)) {
    return Error::make(ErrorCode::InvalidArgument, "HF loader expects a directory: " + path);
  }

  HfConfig cfg;
  Error err = hf_load_config_file(join_path(path, "config.json"), &cfg);
  if (!err.ok()) return err;
  cfg.chat_template = load_optional_chat_template(path);

  std::unordered_map<std::string, std::string> weight_map;
  std::vector<std::string> shard_files;
  err = hf_load_weight_index(path, &weight_map, &shard_files);
  if (!err.ok()) return err;

  auto resolve_shard = [&](const std::string& tensor) -> std::string {
    auto it = weight_map.find(tensor);
    if (it == weight_map.end()) return {};
    return join_path(path, it->second);
  };

  auto has_tensor = [&](const std::string& name) -> bool {
    return weight_map.find(name) != weight_map.end();
  };

  const char* emb_name = nullptr;
  if (has_tensor("model.embed_tokens.weight")) {
    emb_name = "model.embed_tokens.weight";
  } else if (has_tensor("embed_tokens.weight")) {
    emb_name = "embed_tokens.weight";
  }
  const char* lm_name = nullptr;
  if (has_tensor("lm_head.weight")) {
    lm_name = "lm_head.weight";
  }
  // Tied embeddings: missing lm_head (or config tie) → reuse embed_tokens.
  const bool tied_embeddings = (lm_name == nullptr && emb_name != nullptr) ||
                               (cfg.tie_word_embeddings && lm_name == nullptr);
  if (tied_embeddings) {
    lm_name = emb_name;
  }
  const char* norm_name = nullptr;
  if (has_tensor("model.norm.weight")) {
    norm_name = "model.norm.weight";
  } else if (has_tensor("norm.weight")) {
    norm_name = "norm.weight";
  }

  if (emb_name == nullptr || lm_name == nullptr) {
    return Error::make(ErrorCode::InvalidArgument,
                       "HF model missing embed_tokens / lm_head weights under: " + path);
  }
  if (!has_tensor("model.layers.0.self_attn.q_proj.weight")) {
    return Error::make(ErrorCode::InvalidArgument,
                       "HF model missing model.layers.0.self_attn.q_proj.weight under: " + path);
  }

  ShardCache cache;
  auto tensor_shape = [&](const std::string& name, Shape* out_shape) -> Error {
    const std::string shard = resolve_shard(name);
    if (shard.empty()) {
      return Error::make(ErrorCode::NotFound, "tensor not in weight map: " + name);
    }
    Error e = cache.ensure(shard);
    if (!e.ok()) return e;
    const SafetensorsTensorInfo* ti = cache.find(shard, name);
    if (ti == nullptr) {
      return Error::make(ErrorCode::NotFound, "tensor not in shard header: " + name);
    }
    *out_shape = Shape{ti->shape};
    return Error::success();
  };

  Shape emb_shape;
  err = tensor_shape(emb_name, &emb_shape);
  if (!err.ok()) return err;
  Shape lm_shape;
  err = tensor_shape(lm_name, &lm_shape);
  if (!err.ok()) return err;

  const std::int64_t dim = cfg.hidden_size;
  const std::int64_t vocab = cfg.vocab_size;
  const std::int64_t n_heads = cfg.num_attention_heads;
  const std::int64_t n_kv_heads = cfg.num_key_value_heads;
  const std::int64_t head_dim = cfg.head_dim;
  const std::int64_t q_dim = n_heads * head_dim;
  const std::int64_t kv_dim = n_kv_heads * head_dim;
  const std::int64_t inter_dim = cfg.intermediate_size;
  const float rope_theta = cfg.rope_theta;
  const int rope_mode = cfg.rope_mode;
  const float rms_eps = cfg.rms_norm_eps;
  std::int64_t n_layers = apply_layer_cap(cfg.num_hidden_layers);
  const std::string chat_tpl =
      detect_chat_template_family(cfg.chat_template, cfg.architecture);

  ir::GraphBuilder b("hf_transformer");
  b.set_producer("uaii-hf-loader");
  b.set_metadata("source_format", "huggingface");
  b.set_metadata("source_path", path);
  b.set_metadata("architecture", cfg.architecture);
  if (!cfg.model_type.empty()) b.set_metadata("model_type", cfg.model_type);
  if (tied_embeddings) b.set_metadata("tied_embeddings", "1");

  log::info("hf") << "transformer import arch=" << cfg.architecture << " layers=" << n_layers
                  << " heads=" << n_heads << " kv_heads=" << n_kv_heads;

  TensorId tokens = b.add_tensor("tokens", DType::F32, Shape{{1, 1}});
  const std::string emb_ref = resolve_shard(emb_name) + "#" + emb_name;
  TensorId emb_w = b.add_weight(emb_name, DType::F32, emb_shape, emb_ref);
  TensorId hidden = b.add_tensor("hidden", DType::F32, Shape{{1, dim}});
  b.add_node("embed", "Embedding", "1", {tokens, emb_w}, {hidden});

  TensorId features = hidden;

  auto add_weight_ref = [&](const std::string& name, TensorId* out_id) -> Error {
    if (!has_tensor(name)) {
      *out_id = 0;
      return Error::success();
    }
    Shape shape;
    Error e = tensor_shape(name, &shape);
    if (!e.ok()) return e;
    const std::string ref = resolve_shard(name) + "#" + name;
    *out_id = b.add_weight(name, DType::F32, shape, ref);
    return Error::success();
  };

  for (std::int64_t li = 0; li < n_layers; ++li) {
    const std::string hf = "model.layers." + std::to_string(li) + ".";
    const std::string pfx = "l" + std::to_string(li) + ".";

    TensorId attn_norm = 0;
    TensorId wq = 0;
    TensorId wk = 0;
    TensorId wv = 0;
    TensorId wo = 0;
    TensorId ffn_norm = 0;
    TensorId w_gate = 0;
    TensorId w_up = 0;
    TensorId w_down = 0;

    err = add_weight_ref(hf + "input_layernorm.weight", &attn_norm);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "self_attn.q_proj.weight", &wq);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "self_attn.k_proj.weight", &wk);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "self_attn.v_proj.weight", &wv);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "self_attn.o_proj.weight", &wo);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "post_attention_layernorm.weight", &ffn_norm);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "mlp.gate_proj.weight", &w_gate);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "mlp.up_proj.weight", &w_up);
    if (!err.ok()) return err;
    err = add_weight_ref(hf + "mlp.down_proj.weight", &w_down);
    if (!err.ok()) return err;

    if (!attn_norm || !wq || !wk || !wv || !wo) {
      return Error::make(ErrorCode::InvalidArgument,
                         "incomplete attention weights for layer " + std::to_string(li));
    }

    TensorId n1 = b.add_tensor(pfx + "n1", DType::F32, Shape{{1, dim}});
    b.add_node(pfx + "rms1", "RMSNorm", "1", {features, attn_norm}, {n1},
               {ir::make_float_attr("eps", rms_eps)});

    TensorId q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, q_dim}});
    TensorId k = b.add_tensor(pfx + "k", DType::F32, Shape{{1, kv_dim}});
    TensorId v = b.add_tensor(pfx + "v", DType::F32, Shape{{1, kv_dim}});
    b.add_node(pfx + "q_mm", "MatMul", "1", {n1, wq}, {q},
               {ir::make_bool_attr("transpose_b", true)});
    b.add_node(pfx + "k_mm", "MatMul", "1", {n1, wk}, {k},
               {ir::make_bool_attr("transpose_b", true)});
    b.add_node(pfx + "v_mm", "MatMul", "1", {n1, wv}, {v},
               {ir::make_bool_attr("transpose_b", true)});

    TensorId q_rope = b.add_tensor(pfx + "q_rope", DType::F32, Shape{{1, q_dim}});
    TensorId k_rope = b.add_tensor(pfx + "k_rope", DType::F32, Shape{{1, kv_dim}});
    b.add_node(pfx + "q_rope", "RoPE", "1", {q}, {q_rope},
               {ir::make_float_attr("theta", rope_theta), ir::make_int_attr("head_dim", head_dim),
                ir::make_int_attr("mode", rope_mode)});
    b.add_node(pfx + "k_rope", "RoPE", "1", {k}, {k_rope},
               {ir::make_float_attr("theta", rope_theta), ir::make_int_attr("head_dim", head_dim),
                ir::make_int_attr("mode", rope_mode)});

    TensorId attn = b.add_tensor(pfx + "attn", DType::F32, Shape{{1, q_dim}});
    b.add_node(pfx + "attn", "Attention", "1", {q_rope, k_rope, v}, {attn},
               {ir::make_int_attr("num_heads", n_heads), ir::make_int_attr("kv_heads", n_kv_heads),
                ir::make_bool_attr("causal", true), ir::make_bool_attr("use_kv_cache", true),
                ir::make_int_attr("layer_id", li)});

    TensorId attn_o = b.add_tensor(pfx + "attn_o", DType::F32, Shape{{1, dim}});
    b.add_node(pfx + "o_proj", "MatMul", "1", {attn, wo}, {attn_o},
               {ir::make_bool_attr("transpose_b", true)});

    TensorId resid1 = b.add_tensor(pfx + "resid1", DType::F32, Shape{{1, dim}});
    b.add_node(pfx + "add1", "Add", "1", {features, attn_o}, {resid1});

    if (!ffn_norm || !w_gate || !w_up || !w_down) {
      return Error::make(ErrorCode::InvalidArgument,
                         "incomplete MLP/SwiGLU weights for layer " + std::to_string(li));
    }

    TensorId n2 = b.add_tensor(pfx + "n2", DType::F32, Shape{{1, dim}});
    b.add_node(pfx + "rms2", "RMSNorm", "1", {resid1, ffn_norm}, {n2},
               {ir::make_float_attr("eps", rms_eps)});

    TensorId gate = b.add_tensor(pfx + "gate", DType::F32, Shape{{1, inter_dim}});
    TensorId up = b.add_tensor(pfx + "up", DType::F32, Shape{{1, inter_dim}});
    b.add_node(pfx + "ffn_gate", "MatMul", "1", {n2, w_gate}, {gate},
               {ir::make_bool_attr("transpose_b", true)});
    b.add_node(pfx + "ffn_up", "MatMul", "1", {n2, w_up}, {up},
               {ir::make_bool_attr("transpose_b", true)});
    TensorId gate_act = b.add_tensor(pfx + "gate_act", DType::F32, Shape{{1, inter_dim}});
    b.add_node(pfx + "silu", "Silu", "1", {gate}, {gate_act});
    TensorId ff_hid = b.add_tensor(pfx + "ff_hid", DType::F32, Shape{{1, inter_dim}});
    b.add_node(pfx + "ff_mul", "Mul", "1", {gate_act, up}, {ff_hid});
    TensorId ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
    b.add_node(pfx + "ffn_down", "MatMul", "1", {ff_hid, w_down}, {ff_out},
               {ir::make_bool_attr("transpose_b", true)});
    TensorId resid2 = b.add_tensor(pfx + "resid2", DType::F32, Shape{{1, dim}});
    b.add_node(pfx + "add2", "Add", "1", {resid1, ff_out}, {resid2});
    features = resid2;
  }

  if (norm_name != nullptr) {
    Shape nshape;
    err = tensor_shape(norm_name, &nshape);
    if (!err.ok()) return err;
    TensorId nw =
        b.add_weight(norm_name, DType::F32, nshape, resolve_shard(norm_name) + "#" + norm_name);
    TensorId nout = b.add_tensor("normed", DType::F32, Shape{{1, dim}});
    b.add_node("rms_out", "RMSNorm", "1", {features, nw}, {nout},
               {ir::make_float_attr("eps", rms_eps)});
    features = nout;
  }

  TensorId lm_w = tied_embeddings
                      ? emb_w
                      : b.add_weight(lm_name, DType::F32, lm_shape,
                                     resolve_shard(lm_name) + "#" + lm_name);
  TensorId logits = b.add_tensor("logits", DType::F32, Shape{{1, vocab}});
  b.add_node("lm_head", "MatMul", "1", {features, lm_w}, {logits},
             {ir::make_bool_attr("transpose_b", true)});
  TensorId probs = b.add_tensor("probs", DType::F32, Shape{{1, vocab}});
  b.add_node("softmax", "Softmax", "1", {logits}, {probs}, {ir::make_int_attr("axis", -1)});
  b.set_inputs({tokens}).set_outputs({probs});

  b.set_metadata("n_layers", std::to_string(n_layers));
  b.set_metadata("n_heads", std::to_string(n_heads));
  b.set_metadata("kv_heads", std::to_string(n_kv_heads));
  b.set_metadata("head_dim", std::to_string(head_dim));
  b.set_metadata("q_dim", std::to_string(q_dim));
  b.set_metadata("kv_dim", std::to_string(kv_dim));
  b.set_metadata("rope_theta", std::to_string(rope_theta));
  b.set_metadata("rope_mode", std::to_string(rope_mode));
  b.set_metadata("chat_template", chat_tpl);
  b.set_metadata("embedding_length", std::to_string(dim));
  b.set_metadata("rms_eps", std::to_string(rms_eps));
  b.set_metadata("max_context", std::to_string(cfg.max_position_embeddings));
  b.set_metadata("context_length", std::to_string(cfg.max_position_embeddings));

  *out_graph = b.build();
  log::info("hf") << "loaded " << weight_map.size() << " tensors from " << path
                  << " graph=" << out_graph->name;
  return Error::success();
}

}  // namespace loaders
}  // namespace uaii
