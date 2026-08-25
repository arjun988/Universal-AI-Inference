#include "uaii/loaders/hf.hpp"

#include "uaii/core/log.hpp"
#include "uaii/ir/attribute.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/json.hpp"
#include "uaii/loaders/safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace uaii {
namespace loaders {
namespace {

namespace fs = std::filesystem;
namespace json = uaii::ir::json;

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string lower_ascii(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

bool arch_denied(const std::string& arch) {
  const std::string a = lower_ascii(arch);
  static const char* kDeny[] = {
      "vision",     "audio",      "speech",     "whisper", "clip",     "blip",
      "encoderdecoder", "seq2seq", "t5encoder", "t5for",  "wav2vec",  "hubert",
      "vitmodel",   "mamba",      "rwkv",       "ssm",     "jamba",    "falconmamba",
      "pixtral",    "llava",      "qwen2vl",    "qwen2_5_vl", "idefics", "fuyu",
      "kosmos",     "bertmodel",  "bertfor",    "albert",  "electra"};
  for (const char* d : kDeny) {
    if (a.find(d) != std::string::npos) return true;
  }
  if (a.find("encoder") != std::string::npos && a.find("causal") == std::string::npos) {
    return true;
  }
  return false;
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

int rope_mode_for_arch(const std::string& arch) {
  const std::string a = lower_ascii(arch);
  if (a.find("gpt2") != std::string::npos || a.find("gpt-2") != std::string::npos) return -1;
  if (a.find("qwen") != std::string::npos || a.find("gemma") != std::string::npos ||
      a.find("phi") != std::string::npos || a.find("olmo") != std::string::npos ||
      a.find("stablelm") != std::string::npos || a.find("neox") != std::string::npos ||
      a.find("falcon") != std::string::npos) {
    return 2;
  }
  return 0;
}

bool is_gpt_oss_name(const std::string& architecture, const std::string& model_type) {
  const std::string a = lower_ascii(architecture + " " + model_type);
  return a.find("gpt_oss") != std::string::npos || a.find("gpt-oss") != std::string::npos ||
         a.find("gptoss") != std::string::npos;
}

bool is_known_rope_scaling(const std::string& type) {
  return type.empty() || type == "none" || type == "default" || type == "linear" ||
         type == "dynamic" || type == "llama3" || type == "yarn" || type == "longrope" ||
         type == "su";
}

Error parse_float_array(const json::Value* v, std::vector<double>* out) {
  if (v == nullptr || !v->is_array() || out == nullptr) return Error::success();
  out->clear();
  for (const auto& e : v->arr) {
    if (!e.is_number()) {
      return Error::make(ErrorCode::InvalidArgument, "rope factor array must be numeric");
    }
    out->push_back(e.number);
  }
  return Error::success();
}

Error parse_rope_scaling(const json::Object& root, HfConfig* out) {
  const json::Value* rs = json::get(root, "rope_scaling");
  if (rs == nullptr || rs->is_null()) return Error::success();
  if (rs->is_string()) {
    out->rope_scaling_type = lower_ascii(rs->str);
    if (!is_known_rope_scaling(out->rope_scaling_type)) {
      return Error::make(ErrorCode::NotImplemented,
                         "unsupported rope_scaling.type=" + rs->str);
    }
    if (out->rope_scaling_type == "su" || out->rope_scaling_type == "longrope") {
      return Error::make(ErrorCode::NotImplemented,
                         "unsupported rope_scaling.type=" + out->rope_scaling_type +
                             " (need short_factor/long_factor arrays)");
    }
    return Error::success();
  }
  if (!rs->is_object()) {
    return Error::make(ErrorCode::InvalidArgument, "rope_scaling must be an object or string");
  }
  const json::Value* typ = find_key(rs->obj, {"rope_type", "type"});
  if (typ && typ->is_string()) {
    out->rope_scaling_type = lower_ascii(typ->str);
  }
  if (!is_known_rope_scaling(out->rope_scaling_type)) {
    const std::string raw = (typ && typ->is_string()) ? typ->str : out->rope_scaling_type;
    return Error::make(ErrorCode::NotImplemented, "unsupported rope_scaling.type=" + raw);
  }
  if (const json::Value* f = json::get(rs->obj, "factor"); f && f->is_number() && f->number > 0.0) {
    out->rope_scale_factor = static_cast<float>(f->number);
  }
  if (const json::Value* lf = json::get(rs->obj, "low_freq_factor");
      lf && lf->is_number() && lf->number > 0.0) {
    out->rope_low_freq_factor = static_cast<float>(lf->number);
  }
  if (const json::Value* hf = json::get(rs->obj, "high_freq_factor");
      hf && hf->is_number() && hf->number > 0.0) {
    out->rope_high_freq_factor = static_cast<float>(hf->number);
  }
  if (const json::Value* om = find_key(rs->obj, {"original_max_position_embeddings"})) {
    if (om->is_number() && om->number > 0.0) {
      out->rope_original_max_pos = static_cast<float>(om->number);
    }
  }
  if (const json::Value* af = find_key(rs->obj, {"attention_factor", "attn_factor"})) {
    if (af->is_number() && af->number > 0.0) {
      out->rope_attn_factor = static_cast<float>(af->number);
    }
  }
  Error err = parse_float_array(find_key(rs->obj, {"short_factor"}), &out->rope_short_factor);
  if (!err.ok()) return err;
  err = parse_float_array(find_key(rs->obj, {"long_factor"}), &out->rope_long_factor);
  if (!err.ok()) return err;
  if (out->rope_scaling_type == "su" || out->rope_scaling_type == "longrope") {
    if (out->rope_short_factor.empty() && out->rope_long_factor.empty()) {
      return Error::make(ErrorCode::NotImplemented,
                         "unsupported rope_scaling.type=" + out->rope_scaling_type +
                             " (need short_factor/long_factor arrays)");
    }
  }
  return Error::success();
}

void parse_quantization_config(const json::Object& root, HfConfig* out) {
  const json::Value* qc = json::get(root, "quantization_config");
  if (qc == nullptr) return;
  if (qc->is_string()) {
    out->quant_method = lower_ascii(qc->str);
  } else if (qc->is_object()) {
    if (const json::Value* m = find_key(qc->obj, {"quant_method", "quant_type", "bits"})) {
      if (m->is_string()) out->quant_method = lower_ascii(m->str);
    }
  }
  const std::string blob = out->quant_method + " " + lower_ascii(out->architecture) + " " +
                           lower_ascii(out->model_type);
  if (out->quant_method.find("mxfp4") != std::string::npos ||
      blob.find("mxfp4") != std::string::npos) {
    out->mxfp4 = true;
  }
}

bool quant_method_blocked(const std::string& method) {
  const std::string m = lower_ascii(method);
  return m.find("gptq") != std::string::npos || m.find("awq") != std::string::npos ||
         m.find("bitsandbytes") != std::string::npos || m.find("bnb") != std::string::npos ||
         m.find("hqq") != std::string::npos || m.find("compressed") != std::string::npos ||
         m.find("fp8") != std::string::npos || m.find("f8") != std::string::npos ||
         m.find("eetq") != std::string::npos || m.find("aqlm") != std::string::npos;
}

Error map_hidden_act(const std::string& raw, std::string* op) {
  const std::string a = lower_ascii(raw);
  if (a.empty() || a == "silu" || a == "swish" || a == "silu_pytorch") {
    *op = "Silu";
    return Error::success();
  }
  if (a.find("gelu") != std::string::npos) {
    *op = "Gelu";
    return Error::success();
  }
  if (a == "relu") {
    *op = "Relu";
    return Error::success();
  }
  if (a == "relu2") {
    *op = "Relu2";
    return Error::success();
  }
  return Error::make(ErrorCode::NotImplemented, "unsupported hidden_act=" + raw);
}

std::vector<ir::Attribute> make_rms_attrs(const HfConfig& cfg) {
  std::vector<ir::Attribute> attrs;
  attrs.push_back(ir::make_float_attr("eps", cfg.rms_norm_eps));
  if (cfg.rms_offset != 0.f) {
    attrs.push_back(ir::make_float_attr("offset", cfg.rms_offset));
  }
  return attrs;
}

std::vector<ir::Attribute> make_rope_attrs(const HfConfig& cfg, std::int64_t head_dim) {
  std::vector<ir::Attribute> attrs;
  attrs.push_back(ir::make_float_attr("theta", cfg.rope_theta));
  attrs.push_back(ir::make_int_attr("head_dim", head_dim));
  attrs.push_back(ir::make_int_attr("mode", cfg.rope_mode));
  if (!cfg.rope_scaling_type.empty()) {
    attrs.push_back(ir::make_string_attr("scale_type", cfg.rope_scaling_type));
  }
  attrs.push_back(ir::make_float_attr("scale_factor", cfg.rope_scale_factor));
  attrs.push_back(ir::make_float_attr("low_freq_factor", cfg.rope_low_freq_factor));
  attrs.push_back(ir::make_float_attr("high_freq_factor", cfg.rope_high_freq_factor));
  const float old_ctx = cfg.rope_original_max_pos > 0.f
                            ? cfg.rope_original_max_pos
                            : static_cast<float>(cfg.max_position_embeddings);
  attrs.push_back(ir::make_float_attr("old_context_len", old_ctx > 0.f ? old_ctx : 8192.f));
  attrs.push_back(ir::make_float_attr("attn_factor", cfg.rope_attn_factor));
  if (cfg.rotary_dim > 0) {
    attrs.push_back(ir::make_int_attr("rotary_dim", cfg.rotary_dim));
  }
  if (!cfg.rope_short_factor.empty()) {
    attrs.push_back(ir::make_float_array_attr("short_factor", cfg.rope_short_factor));
  }
  if (!cfg.rope_long_factor.empty()) {
    attrs.push_back(ir::make_float_array_attr("long_factor", cfg.rope_long_factor));
  }
  return attrs;
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
  if (a.find("mistral") != std::string::npos || a.find("mixtral") != std::string::npos) {
    return "mistral";
  }
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

  const SafetensorsTensorInfo* find(const std::string& abs_path, const std::string& tensor) {
    auto it = files.find(abs_path);
    if (it == files.end()) return nullptr;
    for (const auto& t : it->second.tensors) {
      if (t.name == tensor) return &t;
    }
    return nullptr;
  }
};

const char* kLayerPrefixes[] = {
    "model.layers.", "model.decoder.layers.", "model.model.layers.", "transformer.h.",
    "transformer.layers.", "gpt_neox.layers.", "transformer.blocks.", "h.",
};

struct DecodedLayout {
  std::string prefix;
  std::string embed;
  std::string lm_head;
  std::string out_norm;
  std::string out_norm_bias;
  enum Attn { Split, Fused } attn = Split;
  bool conv1d = false;  // GPT-2 Conv1D: weight [in, out]
  bool layernorm = false;
  bool rope = true;
  bool gelu = false;
  bool mixtral = false;
  std::string n1, n1b, n2, n2b;
  std::string q, k, v, o, qkv;
  std::string qb, kb, vb, ob, qkvb;
  std::string gate, up, down, fc, proj;
  std::string moe_gate;
  std::string expert_prefix;  // "block_sparse_moe.experts." or "mlp.experts."
  int expert_w_style = 0;  // 0=w1/w2/w3  1=gate/up/down
  std::string q_norm, k_norm;
  std::string post_attn, post_mlp;
  std::string wpe;
  std::string shared_gate, shared_up, shared_down, shared_expert_gate;
};

std::string layer_name(const DecodedLayout& L, std::int64_t li, const std::string& rest) {
  return L.prefix + std::to_string(li) + "." + rest;
}

std::string first_s(const std::unordered_map<std::string, std::string>& m,
                    std::initializer_list<const char*> names) {
  for (const char* n : names) {
    if (m.find(n) != m.end()) return std::string(n);
  }
  return {};
}

bool pick_layout(const std::unordered_map<std::string, std::string>& wm, DecodedLayout* out) {
  if (!out) return false;
  *out = DecodedLayout{};
  out->embed = first_s(wm, {"model.embed_tokens.weight", "embed_tokens.weight",
                            "transformer.wte.weight", "gpt_neox.embed_in.weight",
                            "model.tok_embeddings.weight", "wte.weight", "model.wte.weight",
                            "transformer.word_embeddings.weight"});
  out->wpe = first_s(wm, {"transformer.wpe.weight", "wpe.weight", "model.wpe.weight",
                         "gpt_neox.embed_positions.weight", "model.embed_positions.weight"});
  out->lm_head = first_s(wm, {"lm_head.weight", "embed_out.weight", "output.weight"});
  out->out_norm = first_s(wm, {"model.norm.weight", "norm.weight", "transformer.ln_f.weight",
                               "gpt_neox.final_layer_norm.weight", "model.final_layernorm.weight",
                               "transformer.final_layernorm.weight", "ln_f.weight"});
  out->out_norm_bias = first_s(wm, {"model.norm.bias", "transformer.ln_f.bias",
                                    "gpt_neox.final_layer_norm.bias", "ln_f.bias"});

  for (const char* pfx : kLayerPrefixes) {
    const std::string p0 = std::string(pfx) + "0.";
    auto has = [&](const std::string& s) { return wm.find(p0 + s) != wm.end(); };

    if (!has("self_attn.q_proj.weight") && !has("self_attn.qkv_proj.weight") &&
        !has("self_attn.W_pack.weight") && !has("attn.c_attn.weight") &&
        !has("attention.query_key_value.weight") && !has("self_attn.query_key_value.weight") &&
        !has("attn.q_proj.weight")) {
      continue;
    }
    out->prefix = pfx;

    if (has("self_attn.q_proj.weight")) {
      out->attn = DecodedLayout::Split;
      out->q = "self_attn.q_proj.weight";
      out->k = "self_attn.k_proj.weight";
      out->v = "self_attn.v_proj.weight";
      out->o = has("self_attn.o_proj.weight") ? "self_attn.o_proj.weight"
                                              : "self_attn.dense.weight";
      out->qb = has("self_attn.q_proj.bias") ? "self_attn.q_proj.bias" : "";
      out->kb = has("self_attn.k_proj.bias") ? "self_attn.k_proj.bias" : "";
      out->vb = has("self_attn.v_proj.bias") ? "self_attn.v_proj.bias" : "";
      out->ob = has("self_attn.o_proj.bias") ? "self_attn.o_proj.bias" : "";
      if (has("self_attn.q_norm.weight")) out->q_norm = "self_attn.q_norm.weight";
      if (has("self_attn.k_norm.weight")) out->k_norm = "self_attn.k_norm.weight";
    } else if (has("attn.q_proj.weight")) {
      out->attn = DecodedLayout::Split;
      out->q = "attn.q_proj.weight";
      out->k = "attn.k_proj.weight";
      out->v = "attn.v_proj.weight";
      out->o = has("attn.o_proj.weight") ? "attn.o_proj.weight" : "attn.dense.weight";
    } else if (has("attn.c_attn.weight")) {
      out->attn = DecodedLayout::Fused;
      out->conv1d = true;
      out->qkv = "attn.c_attn.weight";
      out->qkvb = has("attn.c_attn.bias") ? "attn.c_attn.bias" : "";
      out->o = "attn.c_proj.weight";
      out->ob = has("attn.c_proj.bias") ? "attn.c_proj.bias" : "";
      out->rope = false;
      out->layernorm = true;
      out->gelu = true;
    } else {
      out->attn = DecodedLayout::Fused;
      if (has("self_attn.qkv_proj.weight")) {
        out->qkv = "self_attn.qkv_proj.weight";
        out->qkvb = has("self_attn.qkv_proj.bias") ? "self_attn.qkv_proj.bias" : "";
        out->o = has("self_attn.o_proj.weight") ? "self_attn.o_proj.weight"
                                                : "self_attn.dense.weight";
      } else if (has("self_attn.W_pack.weight")) {
        out->qkv = "self_attn.W_pack.weight";
        out->o = "self_attn.o_proj.weight";
      } else if (has("attention.query_key_value.weight")) {
        out->qkv = "attention.query_key_value.weight";
        out->qkvb = has("attention.query_key_value.bias") ? "attention.query_key_value.bias" : "";
        out->o = "attention.dense.weight";
        out->ob = has("attention.dense.bias") ? "attention.dense.bias" : "";
        out->layernorm = true;
      } else {
        out->qkv = "self_attn.query_key_value.weight";
        out->o = has("self_attn.dense.weight") ? "self_attn.dense.weight"
                                               : "self_attn.o_proj.weight";
      }
    }

    if (has("input_layernorm.weight")) {
      out->n1 = "input_layernorm.weight";
      out->n1b = has("input_layernorm.bias") ? "input_layernorm.bias" : "";
    } else if (has("ln_1.weight")) {
      out->n1 = "ln_1.weight";
      out->n1b = has("ln_1.bias") ? "ln_1.bias" : "";
      out->layernorm = true;
    } else if (has("attention_norm.weight")) {
      out->n1 = "attention_norm.weight";
    }

    if (has("pre_feedforward_layernorm.weight")) {
      out->n2 = "pre_feedforward_layernorm.weight";
      out->n2b = has("pre_feedforward_layernorm.bias") ? "pre_feedforward_layernorm.bias" : "";
      if (has("post_attention_layernorm.weight")) {
        out->post_attn = "post_attention_layernorm.weight";
      }
      if (has("post_feedforward_layernorm.weight")) {
        out->post_mlp = "post_feedforward_layernorm.weight";
      }
    } else if (has("post_attention_layernorm.weight")) {
      out->n2 = "post_attention_layernorm.weight";
      out->n2b = has("post_attention_layernorm.bias") ? "post_attention_layernorm.bias" : "";
    } else if (has("ln_2.weight")) {
      out->n2 = "ln_2.weight";
      out->n2b = has("ln_2.bias") ? "ln_2.bias" : "";
      out->layernorm = true;
    } else if (has("ffn_norm.weight")) {
      out->n2 = "ffn_norm.weight";
    }

    if (has("mlp.shared_expert.gate_proj.weight")) {
      out->shared_gate = "mlp.shared_expert.gate_proj.weight";
      out->shared_up = "mlp.shared_expert.up_proj.weight";
      out->shared_down = "mlp.shared_expert.down_proj.weight";
    } else if (has("mlp.shared_experts.gate_proj.weight")) {
      out->shared_gate = "mlp.shared_experts.gate_proj.weight";
      out->shared_up = "mlp.shared_experts.up_proj.weight";
      out->shared_down = "mlp.shared_experts.down_proj.weight";
    }
    if (has("mlp.shared_expert_gate.weight")) {
      out->shared_expert_gate = "mlp.shared_expert_gate.weight";
    }

    if (has("block_sparse_moe.gate.weight") || has("mlp.gate.weight")) {
      out->mixtral = true;
      out->moe_gate = has("block_sparse_moe.gate.weight") ? "block_sparse_moe.gate.weight"
                                                          : "mlp.gate.weight";
      if (wm.find(p0 + "block_sparse_moe.experts.0.w1.weight") != wm.end()) {
        out->expert_prefix = "block_sparse_moe.experts.";
        out->expert_w_style = 0;
      } else if (wm.find(p0 + "mlp.experts.0.gate_proj.weight") != wm.end()) {
        out->expert_prefix = "mlp.experts.";
        out->expert_w_style = 1;
      } else {
        out->expert_prefix = "block_sparse_moe.experts.";
        out->expert_w_style = 1;
      }
    } else if (has("mlp.gate_proj.weight")) {
      out->gate = "mlp.gate_proj.weight";
      out->up = "mlp.up_proj.weight";
      out->down = "mlp.down_proj.weight";
    } else if (has("mlp.c_fc.weight")) {
      out->fc = "mlp.c_fc.weight";
      out->proj = "mlp.c_proj.weight";
      out->gelu = true;
      out->conv1d = true;
    } else if (has("mlp.dense_h_to_4h.weight")) {
      out->fc = "mlp.dense_h_to_4h.weight";
      out->proj = "mlp.dense_4h_to_h.weight";
      out->gelu = true;
    } else if (has("mlp.fc1.weight")) {
      out->fc = "mlp.fc1.weight";
      out->proj = "mlp.fc2.weight";
      out->gelu = true;
    }

    if (!out->n1b.empty() || !out->n2b.empty()) out->layernorm = true;
    return !out->embed.empty() && !out->prefix.empty();
  }
  return false;
}

}  // namespace

bool hf_architecture_supported(const std::string& arch_name) noexcept {
  if (arch_name.empty() || arch_denied(arch_name)) return false;
  if (ends_with(arch_name, "ForCausalLM") || ends_with(arch_name, "LMHeadModel") ||
      ends_with(arch_name, "ForConditionalGeneration")) {
    // ConditionalGeneration often encoder-decoder; still denied by arch_denied (t5/whisper).
    if (ends_with(arch_name, "ForConditionalGeneration") &&
        lower_ascii(arch_name).find("causal") == std::string::npos) {
      return false;
    }
    return true;
  }
  if (arch_name == "MixtralForCausalLM" || arch_name == "Qwen2MoeForCausalLM") return true;
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

  if (const json::Value* enc = json::get(*root, "is_encoder_decoder");
      enc && enc->is_bool() && enc->b) {
    return Error::make(ErrorCode::NotImplemented,
                       "encoder-decoder models are not supported on the HF decoder path");
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
                       "unsupported Hugging Face architecture (need decoder-only CausalLM): " +
                           out->architecture);
  }

  if (const json::Value* mt = json::get(*root, "model_type"); mt && mt->is_string()) {
    out->model_type = mt->str;
    if (arch_denied(out->model_type)) {
      return Error::make(ErrorCode::NotImplemented,
                         "unsupported Hugging Face model_type: " + out->model_type);
    }
  }

  err = require_i64(*root, {"hidden_size", "n_embd", "d_model"}, &out->hidden_size, "hidden_size");
  if (!err.ok()) return err;
  err = require_i64(*root, {"num_hidden_layers", "n_layer", "n_layers"}, &out->num_hidden_layers,
                    "num_hidden_layers");
  if (!err.ok()) return err;
  err = require_i64(*root, {"num_attention_heads", "n_head"}, &out->num_attention_heads,
                    "num_attention_heads");
  if (!err.ok()) return err;
  err = require_i64(*root, {"intermediate_size", "n_inner", "ffn_dim", "n_inner_dim"},
                    &out->intermediate_size, "intermediate_size");
  if (!err.ok()) {
    // GPT-2 often omits n_inner (defaults to 4 * hidden).
    out->intermediate_size = out->hidden_size * 4;
    err = Error::success();
  }
  err = require_i64(*root, {"vocab_size"}, &out->vocab_size, "vocab_size");
  if (!err.ok()) return err;
  err = require_i64(*root, {"max_position_embeddings", "n_positions", "max_sequence_length",
                            "n_ctx"},
                    &out->max_position_embeddings, "max_position_embeddings");
  if (!err.ok()) return err;
  err = require_f32(*root, {"rms_norm_eps", "rms_norm_epsilon", "layer_norm_eps",
                            "layer_norm_epsilon", "layer_norm_eps"},
                    &out->rms_norm_eps, "rms_norm_eps");
  if (!err.ok()) {
    out->rms_norm_eps = 1e-5f;
    err = Error::success();
  }

  if (const json::Value* kv = find_key(*root, {"num_key_value_heads", "n_head_kv",
                                               "num_kv_heads"})) {
    if (!kv->is_number() || kv->number <= 0) {
      return Error::make(ErrorCode::InvalidArgument, "invalid num_key_value_heads");
    }
    out->num_key_value_heads = static_cast<std::int64_t>(kv->number);
  } else {
    out->num_key_value_heads = out->num_attention_heads;
  }

  if (const json::Value* th = find_key(*root, {"rope_theta", "rotary_emb_base"})) {
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

  auto opt_i64 = [&](std::initializer_list<const char*> keys) -> std::int64_t {
    const json::Value* v = find_key(*root, keys);
    if (v && v->is_number() && v->number > 0) return static_cast<std::int64_t>(v->number);
    return 0;
  };
  out->num_experts = opt_i64({"num_local_experts", "num_experts", "n_routed_experts"});
  out->num_experts_per_tok = opt_i64({"num_experts_per_tok", "moe_top_k", "num_experts_per_token"});
  if (out->num_experts_per_tok <= 0 && out->num_experts > 0) out->num_experts_per_tok = 2;

  if (const json::Value* act = find_key(*root, {"hidden_act", "activation_function"})) {
    if (act->is_string()) out->hidden_act = act->str;
  }
  {
    Error aerr = map_hidden_act(out->hidden_act, &out->hidden_act_op);
    if (!aerr.ok()) return aerr;
  }

  parse_quantization_config(*root, out);

  err = parse_rope_scaling(*root, out);
  if (!err.ok()) return err;
  if (out->rope_original_max_pos <= 0.f) {
    out->rope_original_max_pos = static_cast<float>(out->max_position_embeddings);
  }

  if (const json::Value* pe = find_key(*root, {"position_embedding_type"})) {
    if (pe->is_string()) {
      out->position_embedding_type = lower_ascii(pe->str);
      if (out->position_embedding_type.find("alibi") != std::string::npos) {
        return Error::make(ErrorCode::NotImplemented,
                           "unsupported position_embedding_type=alibi");
      }
    }
  }

  const std::string mt = lower_ascii(out->model_type + " " + out->architecture);
  if (mt.find("gemma") != std::string::npos) {
    out->rms_offset = 1.f;
  }

  if (const json::Value* sw = find_key(*root, {"sliding_window"})) {
    if (sw->is_number() && sw->number > 0) {
      out->sliding_window = static_cast<std::int64_t>(sw->number);
    } else if (sw->is_array() && !sw->arr.empty() && sw->arr[0].is_number()) {
      out->sliding_window = static_cast<std::int64_t>(sw->arr[0].number);
    }
  }
  bool use_sw = out->sliding_window > 0;
  if (const json::Value* usw = find_key(*root, {"use_sliding_window"})) {
    if (usw->is_bool()) use_sw = usw->b && out->sliding_window > 0;
  }
  if (!use_sw) out->sliding_window = 0;

  if (const json::Value* lt = json::get(*root, "layer_types");
      lt && lt->is_array() && !lt->arr.empty()) {
    out->layer_sliding_window.assign(static_cast<std::size_t>(out->num_hidden_layers), 0);
    for (std::size_t i = 0; i < lt->arr.size() && i < out->layer_sliding_window.size(); ++i) {
      if (!lt->arr[i].is_string()) continue;
      const std::string t = lower_ascii(lt->arr[i].str);
      if (t.find("sliding") != std::string::npos) {
        out->layer_sliding_window[i] = out->sliding_window > 0 ? out->sliding_window : 4096;
      }
    }
  } else if (out->sliding_window > 0 &&
             (mt.find("gemma2") != std::string::npos || mt.find("gemma3") != std::string::npos)) {
    // Gemma2: layer 0,2,4,... sliding; 1,3,5,... full.
    out->layer_sliding_window.assign(static_cast<std::size_t>(out->num_hidden_layers), 0);
    for (std::int64_t i = 0; i < out->num_hidden_layers; ++i) {
      if ((i % 2) == 0) out->layer_sliding_window[static_cast<std::size_t>(i)] = out->sliding_window;
    }
  }

  auto opt_f = [&](std::initializer_list<const char*> keys) -> float {
    const json::Value* v = find_key(*root, keys);
    if (v && v->is_number() && v->number > 0.0) return static_cast<float>(v->number);
    return 0.f;
  };
  out->attn_logit_softcapping = opt_f({"attn_logit_softcapping"});
  out->final_logit_softcapping = opt_f({"final_logit_softcapping"});

  if (const json::Value* pr = find_key(*root, {"partial_rotary_factor", "rotary_pct"})) {
    if (pr->is_number() && pr->number > 0.0 && pr->number <= 1.0) {
      out->partial_rotary_factor = static_cast<float>(pr->number);
    }
  }
  if (const json::Value* rd = find_key(*root, {"rotary_dim", "rope_dim"})) {
    if (rd->is_number() && rd->number > 0) {
      out->rotary_dim = static_cast<std::int64_t>(rd->number);
    }
  }
  if (out->rotary_dim <= 0 && out->partial_rotary_factor > 0.f &&
      out->partial_rotary_factor < 1.f) {
    out->rotary_dim = static_cast<std::int64_t>(
        static_cast<float>(out->head_dim) * out->partial_rotary_factor);
    if (out->rotary_dim % 2 != 0) out->rotary_dim -= 1;
    if (out->rotary_dim <= 0) out->rotary_dim = 2;
  }

  if (const json::Value* par = json::get(*root, "use_parallel_residual");
      par && par->is_bool()) {
    out->use_parallel_residual = par->b;
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
  return LoaderInfo{"huggingface", "1.1", {}};
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

  if (has_tensor("model.layers.0.self_attn.q_a_proj.weight") ||
      has_tensor("model.layers.0.self_attn.kv_a_proj_with_mqa.weight")) {
    return Error::make(ErrorCode::NotImplemented, "HF MLA; use GGUF");
  }

  if (is_gpt_oss_name(cfg.architecture, cfg.model_type)) {
    if (cfg.mxfp4) {
      return Error::make(ErrorCode::NotImplemented, "mxfp4 convert to BF16/GGUF");
    }
    return Error::make(ErrorCode::NotImplemented, "gpt-oss family handler");
  }

  for (const auto& kv : weight_map) {
    const std::string n = lower_ascii(kv.first);
    if (n.find("qweight") != std::string::npos || n.find("qzeros") != std::string::npos ||
        n.find(".g_idx") != std::string::npos) {
      return Error::make(ErrorCode::NotImplemented, "quantized safetensors");
    }
  }
  if (!cfg.quant_method.empty() && quant_method_blocked(cfg.quant_method)) {
    return Error::make(ErrorCode::NotImplemented, "quantized safetensors");
  }
  if (cfg.mxfp4) {
    return Error::make(ErrorCode::NotImplemented, "mxfp4 convert to BF16/GGUF");
  }

  DecodedLayout L;
  if (!pick_layout(weight_map, &L) || L.embed.empty()) {
    return Error::make(ErrorCode::InvalidArgument,
                       "HF decoder layout not recognized (need CausalLM tensors such as "
                       "model.layers.*.self_attn.q_proj or transformer.h.*.attn.c_attn) under: " +
                           path);
  }

  const std::string fam = hf_family_name(cfg.architecture, cfg.model_type);
  if (fam == "qwen3" && L.q_norm.empty()) {
    return Error::make(ErrorCode::NotImplemented,
                       "QK-norm weights missing (need self_attn.q_norm / k_norm)");
  }
  if ((fam == "gemma2" || fam == "gemma3") && L.post_attn.empty()) {
    return Error::make(ErrorCode::NotImplemented,
                       "Gemma2 extra residual norms missing (pre_feedforward_layernorm)");
  }
  if (fam == "gpt2" && L.wpe.empty()) {
    return Error::make(ErrorCode::NotImplemented, "GPT-2 wpe positional embedding missing");
  }

  const bool tied_embeddings = L.lm_head.empty() || cfg.tie_word_embeddings;
  if (tied_embeddings) L.lm_head = L.embed;
  if (L.lm_head.empty()) {
    return Error::make(ErrorCode::InvalidArgument,
                       "HF model missing embed / lm_head weights under: " + path);
  }
  if (!L.rope) cfg.rope_mode = -1;

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
  err = tensor_shape(L.embed, &emb_shape);
  if (!err.ok()) return err;
  Shape lm_shape;
  err = tensor_shape(L.lm_head, &lm_shape);
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
  const std::string chat_tpl = detect_chat_template_family(cfg.chat_template, cfg.architecture);
  const std::string gate_act_op =
      cfg.hidden_act_op.empty() ? std::string("Silu") : cfg.hidden_act_op;
  std::int64_t n_experts = cfg.num_experts;
  std::int64_t top_k = cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 2;
  if (L.mixtral && n_experts <= 0) {
    if (L.expert_prefix.empty()) L.expert_prefix = "block_sparse_moe.experts.";
    const char* w = L.expert_w_style == 0 ? "w1.weight" : "gate_proj.weight";
    for (std::int64_t e = 0; e < 256; ++e) {
      const std::string t =
          layer_name(L, 0, L.expert_prefix + std::to_string(e) + "." + w);
      if (!has_tensor(t)) {
        n_experts = e;
        break;
      }
    }
  }

  ir::GraphBuilder b("hf_transformer");
  b.set_producer("uaii-hf-loader");
  b.set_metadata("source_format", "huggingface");
  b.set_metadata("source_path", path);
  b.set_metadata("architecture", cfg.architecture);
  b.set_metadata("hf_layout", L.prefix);
  if (!cfg.model_type.empty()) b.set_metadata("model_type", cfg.model_type);
  if (tied_embeddings) b.set_metadata("tied_embeddings", "1");
  if (L.mixtral) b.set_metadata("moe", "1");

  log::info("hf") << "decoder import arch=" << cfg.architecture << " prefix=" << L.prefix
                  << " layers=" << n_layers << " heads=" << n_heads << " kv_heads=" << n_kv_heads
                  << " moe_experts=" << n_experts;

  TensorId tokens = b.add_tensor("tokens", DType::F32, Shape{{1, 1}});
  const std::string emb_ref = resolve_shard(L.embed) + "#" + L.embed;
  TensorId emb_w = b.add_weight(L.embed, DType::F32, emb_shape, emb_ref);
  TensorId hidden = b.add_tensor("hidden", DType::F32, Shape{{1, dim}});
  if (!L.wpe.empty()) {
    TensorId tok_emb = b.add_tensor("tok_emb", DType::F32, Shape{{1, dim}});
    b.add_node("embed", "Embedding", "1", {tokens, emb_w}, {tok_emb});
    Shape wpe_shape;
    err = tensor_shape(L.wpe, &wpe_shape);
    if (!err.ok()) return err;
    TensorId wpe_w = b.add_weight(L.wpe, DType::F32, wpe_shape, resolve_shard(L.wpe) + "#" + L.wpe);
    TensorId pos = b.add_tensor("positions", DType::F32, Shape{{1, 1}});
    b.add_node("positions", "Positions", "1", {tokens}, {pos});
    TensorId pos_emb = b.add_tensor("pos_emb", DType::F32, Shape{{1, dim}});
    b.add_node("wpe", "Embedding", "1", {pos, wpe_w}, {pos_emb});
    b.add_node("embed_add", "Add", "1", {tok_emb, pos_emb}, {hidden});
  } else {
    b.add_node("embed", "Embedding", "1", {tokens, emb_w}, {hidden});
  }
  TensorId features = hidden;

  auto add_weight_ref = [&](const std::string& name, TensorId* out_id,
                            Shape force = {}) -> Error {
    if (name.empty() || !has_tensor(name)) {
      *out_id = 0;
      return Error::success();
    }
    Shape shape;
    Error e = tensor_shape(name, &shape);
    if (!e.ok()) return e;
    if (!force.dims.empty()) shape = force;
    const std::string ref = resolve_shard(name) + "#" + name;
    *out_id = b.add_weight(name, DType::F32, shape, ref);
    return Error::success();
  };

  auto mm = [&](const std::string& name, TensorId x, TensorId w, TensorId y, bool trans_b) {
    std::vector<ir::Attribute> attrs;
    if (trans_b) attrs.push_back(ir::make_bool_attr("transpose_b", true));
    b.add_node(name, "MatMul", "1", {x, w}, {y}, attrs);
  };

  auto maybe_bias = [&](const std::string& node, TensorId x, TensorId bias, TensorId* y,
                        std::int64_t out_dim) {
    if (!bias) {
      *y = x;
      return;
    }
    TensorId t = b.add_tensor(node + "_b", DType::F32, Shape{{1, out_dim}});
    b.add_node(node + "_addb", "Add", "1", {x, bias}, {t});
    *y = t;
  };

  auto add_norm = [&](const std::string& node, TensorId x, TensorId w, TensorId bias, TensorId y) {
    if (L.layernorm && bias) {
      b.add_node(node, "LayerNorm", "1", {x, w, bias}, {y},
                 {ir::make_float_attr("eps", rms_eps)});
    } else {
      b.add_node(node, "RMSNorm", "1", {x, w}, {y}, make_rms_attrs(cfg));
    }
  };

  auto layer_window = [&](std::int64_t li) -> std::int64_t {
    if (li >= 0 && static_cast<std::size_t>(li) < cfg.layer_sliding_window.size()) {
      return cfg.layer_sliding_window[static_cast<std::size_t>(li)];
    }
    return cfg.sliding_window;
  };

  for (std::int64_t li = 0; li < n_layers; ++li) {
    const std::string pfx = "l" + std::to_string(li) + ".";
    auto N = [&](const std::string& rest) { return layer_name(L, li, rest); };

    TensorId attn_norm = 0, attn_norm_b = 0, wq = 0, wk = 0, wv = 0, wo = 0, wqkv = 0;
    TensorId bq = 0, bk = 0, bv = 0, bo = 0, bqkv = 0;
    TensorId ffn_norm = 0, ffn_norm_b = 0;
    err = add_weight_ref(N(L.n1), &attn_norm);
    if (!err.ok()) return err;
    if (!L.n1b.empty()) err = add_weight_ref(N(L.n1b), &attn_norm_b);
    if (!err.ok()) return err;
    err = add_weight_ref(N(L.n2), &ffn_norm);
    if (!err.ok()) return err;
    if (!L.n2b.empty()) err = add_weight_ref(N(L.n2b), &ffn_norm_b);
    if (!err.ok()) return err;
    err = add_weight_ref(N(L.o), &wo);
    if (!err.ok()) return err;
    if (!L.ob.empty()) err = add_weight_ref(N(L.ob), &bo, Shape{{1, dim}});
    if (!err.ok()) return err;

    const bool trans_b = !L.conv1d;

    TensorId n1 = b.add_tensor(pfx + "n1", DType::F32, Shape{{1, dim}});
    if (!attn_norm) {
      return Error::make(ErrorCode::InvalidArgument,
                         "missing attention norm for layer " + std::to_string(li));
    }
    add_norm(pfx + "rms1", features, attn_norm, attn_norm_b, n1);

    TensorId q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, q_dim}});
    TensorId k = b.add_tensor(pfx + "k", DType::F32, Shape{{1, kv_dim}});
    TensorId v = b.add_tensor(pfx + "v", DType::F32, Shape{{1, kv_dim}});

    if (L.attn == DecodedLayout::Split) {
      err = add_weight_ref(N(L.q), &wq);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.k), &wk);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.v), &wv);
      if (!err.ok()) return err;
      if (!L.qb.empty()) err = add_weight_ref(N(L.qb), &bq, Shape{{1, q_dim}});
      if (!err.ok()) return err;
      if (!L.kb.empty()) err = add_weight_ref(N(L.kb), &bk, Shape{{1, kv_dim}});
      if (!err.ok()) return err;
      if (!L.vb.empty()) err = add_weight_ref(N(L.vb), &bv, Shape{{1, kv_dim}});
      if (!err.ok()) return err;
      if (!wq || !wk || !wv || !wo) {
        return Error::make(ErrorCode::InvalidArgument,
                           "incomplete split-QKV weights for layer " + std::to_string(li));
      }
      TensorId q0 = b.add_tensor(pfx + "q0", DType::F32, Shape{{1, q_dim}});
      TensorId k0 = b.add_tensor(pfx + "k0", DType::F32, Shape{{1, kv_dim}});
      TensorId v0 = b.add_tensor(pfx + "v0", DType::F32, Shape{{1, kv_dim}});
      mm(pfx + "q_mm", n1, wq, q0, trans_b);
      mm(pfx + "k_mm", n1, wk, k0, trans_b);
      mm(pfx + "v_mm", n1, wv, v0, trans_b);
      maybe_bias(pfx + "q", q0, bq, &q, q_dim);
      maybe_bias(pfx + "k", k0, bk, &k, kv_dim);
      maybe_bias(pfx + "v", v0, bv, &v, kv_dim);
    } else {
      err = add_weight_ref(N(L.qkv), &wqkv);
      if (!err.ok()) return err;
      if (!L.qkvb.empty()) {
        err = add_weight_ref(N(L.qkvb), &bqkv, Shape{{1, q_dim + kv_dim + kv_dim}});
        if (!err.ok()) return err;
      }
      if (!wqkv || !wo) {
        return Error::make(ErrorCode::InvalidArgument,
                           "incomplete fused-QKV weights for layer " + std::to_string(li));
      }
      const std::int64_t packed = q_dim + kv_dim + kv_dim;
      TensorId qkv = b.add_tensor(pfx + "qkv", DType::F32, Shape{{1, packed}});
      mm(pfx + "qkv_mm", n1, wqkv, qkv, trans_b);
      TensorId qkv_b = qkv;
      maybe_bias(pfx + "qkv", qkv, bqkv, &qkv_b, packed);
      b.add_node(pfx + "q_slice", "Slice", "1", {qkv_b}, {q},
                 {ir::make_int_attr("start", 0), ir::make_int_attr("end", q_dim)});
      b.add_node(pfx + "k_slice", "Slice", "1", {qkv_b}, {k},
                 {ir::make_int_attr("start", q_dim), ir::make_int_attr("end", q_dim + kv_dim)});
      b.add_node(pfx + "v_slice", "Slice", "1", {qkv_b}, {v},
                 {ir::make_int_attr("start", q_dim + kv_dim), ir::make_int_attr("end", packed)});
    }

    TensorId q_in = q;
    TensorId k_in = k;
    if (!L.q_norm.empty()) {
      TensorId w_qn = 0, w_kn = 0;
      err = add_weight_ref(N(L.q_norm), &w_qn);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.k_norm), &w_kn);
      if (!err.ok()) return err;
      if (w_qn) {
        TensorId qn = b.add_tensor(pfx + "q_n", DType::F32, Shape{{1, q_dim}});
        b.add_node(pfx + "q_norm", "RMSNorm", "1", {q, w_qn}, {qn}, make_rms_attrs(cfg));
        q_in = qn;
      }
      if (w_kn) {
        TensorId kn = b.add_tensor(pfx + "k_n", DType::F32, Shape{{1, kv_dim}});
        b.add_node(pfx + "k_norm", "RMSNorm", "1", {k, w_kn}, {kn}, make_rms_attrs(cfg));
        k_in = kn;
      }
    }
    if (L.rope && rope_mode >= 0) {
      TensorId q_rope = b.add_tensor(pfx + "q_rope", DType::F32, Shape{{1, q_dim}});
      TensorId k_rope = b.add_tensor(pfx + "k_rope", DType::F32, Shape{{1, kv_dim}});
      b.add_node(pfx + "q_rope", "RoPE", "1", {q_in}, {q_rope}, make_rope_attrs(cfg, head_dim));
      b.add_node(pfx + "k_rope", "RoPE", "1", {k_in}, {k_rope}, make_rope_attrs(cfg, head_dim));
      q_in = q_rope;
      k_in = k_rope;
    }

    std::vector<ir::Attribute> attn_attrs = {
        ir::make_int_attr("num_heads", n_heads), ir::make_int_attr("kv_heads", n_kv_heads),
        ir::make_bool_attr("causal", true), ir::make_bool_attr("use_kv_cache", true),
        ir::make_int_attr("layer_id", li)};
    if (cfg.attn_logit_softcapping > 0.f) {
      attn_attrs.push_back(ir::make_float_attr("logit_softcap", cfg.attn_logit_softcapping));
    }
    const std::int64_t win = layer_window(li);
    if (win > 0) {
      attn_attrs.push_back(ir::make_int_attr("sliding_window", win));
    }
    TensorId attn = b.add_tensor(pfx + "attn", DType::F32, Shape{{1, q_dim}});
    b.add_node(pfx + "attn", "Attention", "1", {q_in, k_in, v}, {attn}, attn_attrs);
    TensorId attn_o0 = b.add_tensor(pfx + "attn_o0", DType::F32, Shape{{1, dim}});
    mm(pfx + "o_proj", attn, wo, attn_o0, trans_b);
    TensorId attn_o = attn_o0;
    maybe_bias(pfx + "o", attn_o0, bo, &attn_o, dim);
    if (!L.post_attn.empty()) {
      TensorId w_pa = 0;
      err = add_weight_ref(N(L.post_attn), &w_pa);
      if (!err.ok()) return err;
      if (w_pa) {
        TensorId pa = b.add_tensor(pfx + "post_attn", DType::F32, Shape{{1, dim}});
        add_norm(pfx + "post_attn_rms", attn_o, w_pa, 0, pa);
        attn_o = pa;
      }
    }
    TensorId resid1 = 0;
    if (!cfg.use_parallel_residual) {
      resid1 = b.add_tensor(pfx + "resid1", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "add1", "Add", "1", {features, attn_o}, {resid1});
    }

    const bool parallel = cfg.use_parallel_residual;
    TensorId mlp_in = parallel ? features : resid1;
    if (ffn_norm) {
      TensorId n2 = b.add_tensor(pfx + "n2", DType::F32, Shape{{1, dim}});
      add_norm(pfx + "rms2", mlp_in, ffn_norm, ffn_norm_b, n2);
      mlp_in = n2;
    }

    TensorId ff_out = 0;
    if (L.mixtral && n_experts > 0) {
      if (L.expert_prefix.empty()) L.expert_prefix = "block_sparse_moe.experts.";
      TensorId gate_w = 0;
      err = add_weight_ref(N(L.moe_gate), &gate_w);
      if (!err.ok()) return err;
      if (!gate_w) {
        return Error::make(ErrorCode::InvalidArgument,
                           "MoE gate missing at layer " + std::to_string(li));
      }
      TensorId router_probs = b.add_tensor(pfx + "moe_probs", DType::F32, Shape{{1, n_experts}});
      TensorId router_top = b.add_tensor(pfx + "moe_top", DType::F32, Shape{{1, 1}});
      b.add_node(pfx + "moe_router", "MoERouter", "1", {mlp_in, gate_w},
                 {router_probs, router_top},
                 {ir::make_int_attr("num_experts", n_experts), ir::make_int_attr("top_k", top_k)});
      std::vector<TensorId> moe_ins = {mlp_in, router_probs};
      for (std::int64_t e = 0; e < n_experts; ++e) {
        std::string gname, uname, dname;
        const std::string ep = L.expert_prefix + std::to_string(e) + ".";
        if (L.expert_w_style == 0) {
          gname = N(ep + "w1.weight");
          uname = N(ep + "w3.weight");
          dname = N(ep + "w2.weight");
        } else {
          gname = N(ep + "gate_proj.weight");
          uname = N(ep + "up_proj.weight");
          dname = N(ep + "down_proj.weight");
        }
        TensorId ge = 0, ue = 0, de = 0;
        err = add_weight_ref(gname, &ge);
        if (!err.ok()) return err;
        err = add_weight_ref(uname, &ue);
        if (!err.ok()) return err;
        err = add_weight_ref(dname, &de);
        if (!err.ok()) return err;
        if (!ge || !ue || !de) {
          return Error::make(ErrorCode::InvalidArgument,
                             "incomplete Mixtral expert " + std::to_string(e) + " layer " +
                                 std::to_string(li));
        }
        moe_ins.push_back(ge);
        moe_ins.push_back(ue);
        moe_ins.push_back(de);
      }
      ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "moe_exps", "MoEExpertsSwiGLU", "1", moe_ins, {ff_out},
                 {ir::make_int_attr("num_experts", n_experts), ir::make_int_attr("top_k", top_k)});
    } else if (!L.gate.empty()) {
      TensorId w_gate = 0, w_up = 0, w_down = 0;
      err = add_weight_ref(N(L.gate), &w_gate);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.up), &w_up);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.down), &w_down);
      if (!err.ok()) return err;
      if (!w_gate || !w_up || !w_down) {
        return Error::make(ErrorCode::InvalidArgument,
                           "incomplete SwiGLU MLP for layer " + std::to_string(li));
      }
      TensorId gate = b.add_tensor(pfx + "gate", DType::F32, Shape{{1, inter_dim}});
      TensorId up = b.add_tensor(pfx + "up", DType::F32, Shape{{1, inter_dim}});
      mm(pfx + "ffn_gate", mlp_in, w_gate, gate, trans_b);
      mm(pfx + "ffn_up", mlp_in, w_up, up, trans_b);
      TensorId gate_act = b.add_tensor(pfx + "gate_act", DType::F32, Shape{{1, inter_dim}});
      b.add_node(pfx + "gate_act", gate_act_op, "1", {gate}, {gate_act});
      TensorId ff_hid = b.add_tensor(pfx + "ff_hid", DType::F32, Shape{{1, inter_dim}});
      b.add_node(pfx + "ff_mul", "Mul", "1", {gate_act, up}, {ff_hid});
      ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
      mm(pfx + "ffn_down", ff_hid, w_down, ff_out, trans_b);
    } else if (!L.fc.empty()) {
      TensorId w_fc = 0, w_proj = 0;
      err = add_weight_ref(N(L.fc), &w_fc);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.proj), &w_proj);
      if (!err.ok()) return err;
      TensorId up = b.add_tensor(pfx + "up", DType::F32, Shape{{1, inter_dim}});
      mm(pfx + "ffn_up", mlp_in, w_fc, up, trans_b);
      TensorId up_act = b.add_tensor(pfx + "up_act", DType::F32, Shape{{1, inter_dim}});
      const std::string dense_act = L.gelu ? "Gelu" : gate_act_op;
      b.add_node(pfx + "ff_act", dense_act, "1", {up}, {up_act});
      ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
      mm(pfx + "ffn_down", up_act, w_proj, ff_out, trans_b);
    }

    if (ff_out && !L.post_mlp.empty()) {
      TensorId w_pm = 0;
      err = add_weight_ref(N(L.post_mlp), &w_pm);
      if (!err.ok()) return err;
      if (w_pm) {
        TensorId pm = b.add_tensor(pfx + "post_mlp", DType::F32, Shape{{1, dim}});
        add_norm(pfx + "post_mlp_rms", ff_out, w_pm, 0, pm);
        ff_out = pm;
      }
    }

    if (ff_out && !L.shared_gate.empty()) {
      TensorId sg = 0, su = 0, sd = 0;
      err = add_weight_ref(N(L.shared_gate), &sg);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.shared_up), &su);
      if (!err.ok()) return err;
      err = add_weight_ref(N(L.shared_down), &sd);
      if (!err.ok()) return err;
      if (sg && su && sd) {
        Shape sg_shape;
        std::int64_t shexp_inter = inter_dim;
        if (tensor_shape(N(L.shared_gate), &sg_shape).ok() && sg_shape.dims.size() >= 2) {
          shexp_inter = sg_shape.dims[0] > 0 ? sg_shape.dims[0] : inter_dim;
        }
        TensorId sh_g = b.add_tensor(pfx + "shexp_gate", DType::F32, Shape{{1, shexp_inter}});
        TensorId sh_u = b.add_tensor(pfx + "shexp_up", DType::F32, Shape{{1, shexp_inter}});
        mm(pfx + "shexp_gate_mm", mlp_in, sg, sh_g, trans_b);
        mm(pfx + "shexp_up_mm", mlp_in, su, sh_u, trans_b);
        TensorId sh_ga = b.add_tensor(pfx + "shexp_gate_act", DType::F32, Shape{{1, shexp_inter}});
        b.add_node(pfx + "shexp_act", gate_act_op, "1", {sh_g}, {sh_ga});
        TensorId sh_hid = b.add_tensor(pfx + "shexp_hid", DType::F32, Shape{{1, shexp_inter}});
        b.add_node(pfx + "shexp_mul", "Mul", "1", {sh_ga, sh_u}, {sh_hid});
        TensorId sh_out = b.add_tensor(pfx + "shexp_out", DType::F32, Shape{{1, dim}});
        mm(pfx + "shexp_down", sh_hid, sd, sh_out, trans_b);
        TensorId merged = b.add_tensor(pfx + "ff_merged", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "ff_add_shexp", "Add", "1", {ff_out, sh_out}, {merged});
        ff_out = merged;
      }
    }

    if (ff_out) {
      TensorId resid2 = b.add_tensor(pfx + "resid2", DType::F32, Shape{{1, dim}});
      if (parallel) {
        TensorId mid = b.add_tensor(pfx + "par_mid", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "add_par_attn", "Add", "1", {features, attn_o}, {mid});
        b.add_node(pfx + "add2", "Add", "1", {mid, ff_out}, {resid2});
      } else {
        b.add_node(pfx + "add2", "Add", "1", {resid1, ff_out}, {resid2});
      }
      features = resid2;
    } else if (parallel) {
      TensorId only = b.add_tensor(pfx + "resid1", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "add1", "Add", "1", {features, attn_o}, {only});
      features = only;
    } else {
      features = resid1;
    }
  }

  if (!L.out_norm.empty()) {
    Shape nshape;
    err = tensor_shape(L.out_norm, &nshape);
    if (!err.ok()) return err;
    TensorId nw =
        b.add_weight(L.out_norm, DType::F32, nshape, resolve_shard(L.out_norm) + "#" + L.out_norm);
    TensorId nb = 0;
    if (!L.out_norm_bias.empty() && has_tensor(L.out_norm_bias)) {
      nb = b.add_weight(L.out_norm_bias, DType::F32, Shape{{1, dim}},
                        resolve_shard(L.out_norm_bias) + "#" + L.out_norm_bias);
    }
    TensorId nout = b.add_tensor("normed", DType::F32, Shape{{1, dim}});
    if (L.layernorm && nb) {
      b.add_node("ln_out", "LayerNorm", "1", {features, nw, nb}, {nout},
                 {ir::make_float_attr("eps", rms_eps)});
    } else {
      b.add_node("rms_out", "RMSNorm", "1", {features, nw}, {nout}, make_rms_attrs(cfg));
    }
    features = nout;
  }

  TensorId lm_w = tied_embeddings
                      ? emb_w
                      : b.add_weight(L.lm_head, DType::F32, lm_shape,
                                     resolve_shard(L.lm_head) + "#" + L.lm_head);
  TensorId logits = b.add_tensor("logits", DType::F32, Shape{{1, vocab}});
  b.add_node("lm_head", "MatMul", "1", {features, lm_w}, {logits},
             {ir::make_bool_attr("transpose_b", true)});
  TensorId logits_in = logits;
  if (cfg.final_logit_softcapping > 0.f) {
    TensorId capped = b.add_tensor("logits_cap", DType::F32, Shape{{1, vocab}});
    b.add_node("logit_softcap", "Softcap", "1", {logits}, {capped},
               {ir::make_float_attr("cap", cfg.final_logit_softcapping)});
    logits_in = capped;
  }
  TensorId probs = b.add_tensor("probs", DType::F32, Shape{{1, vocab}});
  b.add_node("softmax", "Softmax", "1", {logits_in}, {probs}, {ir::make_int_attr("axis", -1)});
  b.set_inputs({tokens}).set_outputs({probs});

  b.set_metadata("n_layers", std::to_string(n_layers));
  b.set_metadata("n_heads", std::to_string(n_heads));
  b.set_metadata("kv_heads", std::to_string(n_kv_heads));
  b.set_metadata("head_dim", std::to_string(head_dim));
  b.set_metadata("q_dim", std::to_string(q_dim));
  b.set_metadata("kv_dim", std::to_string(kv_dim));
  b.set_metadata("rope_theta", std::to_string(rope_theta));
  b.set_metadata("rope_mode", std::to_string(rope_mode));
  if (!cfg.rope_scaling_type.empty()) {
    b.set_metadata("rope_scaling", cfg.rope_scaling_type);
  }
  b.set_metadata("family", hf_family_name(cfg.architecture, cfg.model_type));
  b.set_metadata("chat_template", chat_tpl);
  if (!cfg.chat_template.empty()) {
    b.set_metadata("chat_template_kind", "family_equivalent");
  }
  b.set_metadata("embedding_length", std::to_string(dim));
  b.set_metadata("rms_eps", std::to_string(rms_eps));
  b.set_metadata("max_context", std::to_string(cfg.max_position_embeddings));
  b.set_metadata("context_length", std::to_string(cfg.max_position_embeddings));

  *out_graph = b.build();
  log::info("hf") << "loaded " << weight_map.size() << " tensors from " << path
                  << " graph=" << out_graph->name;
  return Error::success();
}

Error hf_verify_shards_present(const std::string& dir, const std::vector<std::string>& shards) {
  if (dir.empty()) {
    return Error::make(ErrorCode::InvalidArgument, "shard dir empty");
  }
  for (const auto& shard : shards) {
    if (shard.empty()) {
      return Error::make(ErrorCode::InvalidArgument, "empty shard name");
    }
    const fs::path p = fs::path(dir) / shard;
    std::error_code ec;
    if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) {
      return Error::make(ErrorCode::NotFound, "pull incomplete: missing shard " + shard);
    }
    if (fs::file_size(p, ec) == 0 || ec) {
      return Error::make(ErrorCode::NotFound, "pull incomplete: missing shard " + shard);
    }
  }
  return Error::success();
}

std::string hf_family_name(const std::string& architecture, const std::string& model_type) noexcept {
  const std::string a = lower_ascii(architecture + " " + model_type);
  if (a.find("gpt_oss") != std::string::npos || a.find("gpt-oss") != std::string::npos ||
      a.find("gptoss") != std::string::npos) {
    return "gpt_oss";
  }
  if (a.find("gpt2") != std::string::npos || a.find("gpt-2") != std::string::npos) return "gpt2";
  if (a.find("qwen3") != std::string::npos) return "qwen3";
  if (a.find("qwen2_moe") != std::string::npos || a.find("qwen2moe") != std::string::npos) {
    return "qwen2_moe";
  }
  if (a.find("qwen") != std::string::npos) return "qwen2";
  if (a.find("mixtral") != std::string::npos) return "mixtral";
  if (a.find("mistral") != std::string::npos) return "mistral";
  if (a.find("gemma3") != std::string::npos) return "gemma3";
  if (a.find("gemma2") != std::string::npos) return "gemma2";
  if (a.find("gemma") != std::string::npos) return "gemma";
  if (a.find("phi") != std::string::npos) return "phi";
  if (a.find("olmo") != std::string::npos) return "olmo";
  if (a.find("kimi") != std::string::npos || a.find("deepseek") != std::string::npos) {
    return "kimi_mla";
  }
  if (a.find("llama") != std::string::npos || a.find("smollm") != std::string::npos ||
      a.find("granite") != std::string::npos) {
    return "llama";
  }
  if (!model_type.empty()) return lower_ascii(model_type);
  if (!architecture.empty()) return lower_ascii(architecture);
  return "unknown";
}

Error hf_index_unique_shards(const std::string& index_json, std::vector<std::string>* shards) {
  if (shards == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "shards out null");
  }
  shards->clear();
  json::Value root_v;
  Error err = json::parse(index_json, &root_v);
  if (!err.ok()) return err;
  const json::Object* root = nullptr;
  err = json::require_object(root_v, &root, "weight index");
  if (!err.ok()) return err;
  const json::Value* wm = json::get(*root, "weight_map");
  if (wm == nullptr || !wm->is_object()) {
    return Error::make(ErrorCode::InvalidArgument, "weight index missing weight_map");
  }
  std::unordered_map<std::string, bool> seen;
  for (const auto& kv : wm->obj) {
    if (!kv.second.is_string() || kv.second.str.empty()) {
      return Error::make(ErrorCode::InvalidArgument,
                         "invalid weight_map entry for tensor: " + kv.first);
    }
    if (!seen[kv.second.str]) {
      seen[kv.second.str] = true;
      shards->push_back(kv.second.str);
    }
  }
  if (shards->empty()) {
    return Error::make(ErrorCode::InvalidArgument, "empty weight_map");
  }
  return Error::success();
}

std::string hf_format_capability(const HfCapability& cap) {
  std::ostringstream os;
  os << "format: huggingface\n";
  os << "family=" << cap.family << "\n";
  os << "architecture: " << cap.architecture << "\n";
  if (!cap.model_type.empty()) os << "model_type: " << cap.model_type << "\n";
  os << "weights: " << (cap.weight_dtype.empty() ? "unknown" : cap.weight_dtype) << "\n";
  os << "tokenizer: " << (cap.tokenizer_kind.empty() ? "none" : cap.tokenizer_kind) << "\n";
  os << "rope_scaling: " << (cap.rope_scaling_type.empty() ? "none" : cap.rope_scaling_type)
     << "\n";
  os << "moe: " << (cap.moe ? "yes" : "no") << "\n";
  os << "mla: " << (cap.mla ? "yes" : "no") << "\n";
  if (cap.quantized) os << "quantized: yes\n";
  if (cap.mxfp4) os << "mxfp4: yes\n";
  for (const auto& n : cap.notes) os << "note: " << n << "\n";
  if (cap.ok()) {
    os << "status: ok\n";
  } else {
    os << "status: blocked\n";
    for (const auto& b : cap.blocked) os << "blocked: " << b << "\n";
  }
  return os.str();
}

Error hf_inspect_directory(const std::string& dir, HfCapability* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "HfCapability out null");
  }
  *out = HfCapability{};
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    return Error::make(ErrorCode::InvalidArgument, "not a directory: " + dir);
  }
  const std::string cfg_path = join_path(dir, "config.json");
  if (!fs::exists(cfg_path, ec)) {
    return Error::make(ErrorCode::NotFound, "no config.json under: " + dir);
  }

  Error err;
  const std::string text = read_text_file(cfg_path, &err);
  if (!err.ok()) return err;
  json::Value root_v;
  if (json::parse(text, &root_v).ok() && root_v.is_object()) {
    if (const json::Value* arches = json::get(root_v.obj, "architectures");
        arches && arches->is_array() && !arches->arr.empty() && arches->arr[0].is_string()) {
      out->architecture = arches->arr[0].str;
    }
    if (const json::Value* mt = json::get(root_v.obj, "model_type"); mt && mt->is_string()) {
      out->model_type = mt->str;
    }
  }

  HfConfig cfg;
  err = hf_parse_config_json(text, &cfg);
  if (err.ok()) {
    out->architecture = cfg.architecture;
    out->model_type = cfg.model_type;
    out->rope_scaling_type = cfg.rope_scaling_type.empty() ? "none" : cfg.rope_scaling_type;
    out->quantized = !cfg.quant_method.empty() && quant_method_blocked(cfg.quant_method);
    out->mxfp4 = cfg.mxfp4;
    out->moe = cfg.num_experts > 0;
    if (cfg.sliding_window > 0) {
      out->notes.push_back("sliding_window=" + std::to_string(cfg.sliding_window));
    }
    if (cfg.rms_offset != 0.f) {
      out->notes.push_back("rmsnorm offset=1 (Gemma 1+w)");
    }
  } else {
    out->blocked.push_back(err.message());
  }
  out->family = hf_family_name(out->architecture, out->model_type);

  if (fs::exists(fs::path(dir) / "tokenizer.json", ec)) {
    Error te;
    const std::string tj = read_text_file(join_path(dir, "tokenizer.json"), &te);
    std::string kind = "bpe";
    if (te.ok()) {
      json::Value tv;
      if (json::parse(tj, &tv).ok() && tv.is_object()) {
        if (const json::Value* model = json::get(tv.obj, "model"); model && model->is_object()) {
          if (const json::Value* ty = json::get(model->obj, "type"); ty && ty->is_string()) {
            kind = lower_ascii(ty->str);
          }
        }
      }
    }
    out->tokenizer_kind = kind;
    if (kind == "unigram" || kind == "wordpiece" || kind == "wordlevel") {
      out->blocked.push_back(
          "Unigram tokenizer.json is not loaded. Use tokenizer.model (SentencePiece) or a BPE "
          "tokenizer.json");
    }
  } else if (fs::exists(fs::path(dir) / "tokenizer.model", ec)) {
    out->tokenizer_kind = "sentencepiece";
  } else if (fs::exists(fs::path(dir) / "vocab.json", ec) &&
             fs::exists(fs::path(dir) / "merges.txt", ec)) {
    out->tokenizer_kind = "bpe";
  } else {
    out->tokenizer_kind = "none";
  }

  std::unordered_map<std::string, std::string> weight_map;
  std::vector<std::string> shard_files;
  err = hf_load_weight_index(dir, &weight_map, &shard_files);
  if (!err.ok()) {
    out->blocked.push_back(err.message());
    return Error::success();
  }

  auto has_tensor = [&](const std::string& name) { return weight_map.find(name) != weight_map.end(); };
  if (has_tensor("model.layers.0.self_attn.q_a_proj.weight") ||
      has_tensor("model.layers.0.self_attn.kv_a_proj_with_mqa.weight")) {
    out->mla = true;
    out->family = "kimi_mla";
    out->blocked.push_back("HF MLA; use GGUF");
  }

  if (out->family == "qwen3" &&
      weight_map.find("model.layers.0.self_attn.q_norm.weight") == weight_map.end()) {
    out->blocked.push_back("QK-norm weights missing (need self_attn.q_norm / k_norm)");
  }
  if ((out->family == "gemma2" || out->family == "gemma3") &&
      weight_map.find("model.layers.0.pre_feedforward_layernorm.weight") == weight_map.end()) {
    out->blocked.push_back("Gemma2 extra residual norms missing (pre_feedforward_layernorm)");
  }
  if (out->family == "gpt2" && weight_map.find("transformer.wpe.weight") == weight_map.end() &&
      weight_map.find("wpe.weight") == weight_map.end()) {
    out->blocked.push_back("GPT-2 wpe positional embedding missing");
  }

  bool quantized_tensors = false;
  for (const auto& kv : weight_map) {
    const std::string n = lower_ascii(kv.first);
    if (n.find("qweight") != std::string::npos || n.find("qzeros") != std::string::npos) {
      quantized_tensors = true;
      break;
    }
  }
  if (quantized_tensors || out->quantized) {
    out->quantized = true;
    out->blocked.push_back("quantized safetensors");
  }

  std::unordered_set<std::string> dtypes;
  ShardCache cache;
  for (const auto& rel : shard_files) {
    const std::string abs = join_path(dir, rel);
    Error se = cache.ensure(abs);
    if (!se.ok()) {
      out->blocked.push_back(se.message());
      continue;
    }
    auto it = cache.files.find(abs);
    if (it == cache.files.end()) continue;
    for (const auto& t : it->second.tensors) {
      const std::string dt = t.dtype;
      dtypes.insert(dt);
      const std::string dtl = lower_ascii(dt);
      if (dtl.find("f8") != std::string::npos) {
        out->blocked.push_back("safetensors dtype " + dt +
                               " is not loaded. Convert this checkpoint to BF16.");
      }
      if (dtl == "mxfp4" || dtl.find("float4") != std::string::npos) {
        out->mxfp4 = true;
      }
    }
  }
  std::string joined;
  for (const auto& d : dtypes) {
    if (!joined.empty()) joined += ",";
    joined += d;
  }
  out->weight_dtype = joined;

  if (is_gpt_oss_name(out->architecture, out->model_type)) {
    out->family = "gpt_oss";
    if (out->mxfp4) {
      out->blocked.push_back("mxfp4 convert to BF16/GGUF");
    } else {
      out->blocked.push_back("gpt-oss family handler");
    }
  } else if (out->mxfp4) {
    out->blocked.push_back("mxfp4 convert to BF16/GGUF");
  }

  // Deduplicate blocked lines while preserving order.
  std::vector<std::string> uniq;
  std::unordered_set<std::string> seen;
  for (const auto& b : out->blocked) {
    if (seen.insert(b).second) uniq.push_back(b);
  }
  out->blocked = std::move(uniq);
  return Error::success();
}

}  // namespace loaders
}  // namespace uaii
