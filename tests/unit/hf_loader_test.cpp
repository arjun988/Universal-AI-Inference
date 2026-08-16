#include "uaii/ir/graph.hpp"
#include "uaii/loaders/gguf.hpp"
#include "uaii/loaders/hf.hpp"
#include "uaii/loaders/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

namespace fs = std::filesystem;

std::vector<float> fill_f32(std::size_t n, float seed) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = std::sin(seed + static_cast<float>(i) * 0.017f) * 0.1f;
  }
  return v;
}

void put_tensor(std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>>* tensors,
                const std::string& name,
                std::initializer_list<std::int64_t> dims,
                float seed) {
  uaii::Shape shape;
  shape.dims.assign(dims.begin(), dims.end());
  std::size_t n = 1;
  for (std::int64_t d : shape.dims) {
    n *= static_cast<std::size_t>(d);
  }
  (*tensors)[name] = {shape, fill_f32(n, seed)};
}

void test_parse_config() {
  const char* json = R"({
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000,
    "tie_word_embeddings": true
  })";

  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(err.ok(), "hf_parse_config_json ok");
  expect(cfg.architecture == "LlamaForCausalLM", "architecture");
  expect(cfg.model_type == "llama", "model_type");
  expect(cfg.hidden_size == 32, "hidden_size");
  expect(cfg.num_hidden_layers == 2, "num_hidden_layers");
  expect(cfg.num_attention_heads == 4, "num_attention_heads");
  expect(cfg.num_key_value_heads == 2, "num_key_value_heads");
  expect(cfg.intermediate_size == 64, "intermediate_size");
  expect(cfg.vocab_size == 128, "vocab_size");
  expect(cfg.max_position_embeddings == 128, "max_position_embeddings");
  expect(cfg.head_dim == 8, "head_dim derived");
  expect(cfg.rms_norm_eps > 0.f && cfg.rms_norm_eps < 1e-4f, "rms_norm_eps");
  expect(cfg.rope_theta == 10000.f, "rope_theta");
  expect(cfg.tie_word_embeddings, "tie_word_embeddings");
  expect(cfg.rope_mode == 0, "rope_mode llama interleaved");
}

void test_reject_unknown_arch() {
  const char* json = R"({
    "architectures": ["BertModel"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5
  })";
  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(!err.ok(), "reject encoder BertModel");
}

void test_accept_generic_causal() {
  const char* json = R"({
    "architectures": ["TotallyFakeForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5
  })";
  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(err.ok(), "accept generic *ForCausalLM");
}

void test_moe_config_ok() {
  const char* json = R"({
    "architectures": ["MixtralForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "num_local_experts": 4,
    "num_experts_per_tok": 2
  })";
  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(err.ok(), "Mixtral config parses");
  expect(cfg.num_experts == 4, "num_experts");
  expect(cfg.num_experts_per_tok == 2, "top_k");
}

void test_architecture_supported() {
  expect(uaii::loaders::hf_architecture_supported("LlamaForCausalLM"), "llama supported");
  expect(uaii::loaders::hf_architecture_supported("Qwen2ForCausalLM"), "qwen2 supported");
  expect(uaii::loaders::hf_architecture_supported("Gemma2ForCausalLM"), "gemma2 supported");
  expect(uaii::loaders::hf_architecture_supported("GPT2LMHeadModel"), "gpt2 supported");
  expect(uaii::loaders::hf_architecture_supported("MixtralForCausalLM"), "mixtral supported");
  expect(!uaii::loaders::hf_architecture_supported("BertModel"), "bert unsupported");
  expect(!uaii::loaders::hf_architecture_supported(""), "empty unsupported");
}

void test_hf_loader_roundtrip() {
  const fs::path dir = fs::temp_directory_path() / "uaii_hf_loader_test_tiny";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  expect(!ec, "create temp hf dir");

  {
    std::ofstream cfg(dir / "config.json");
    cfg << R"({
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 32,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "intermediate_size": 64,
  "vocab_size": 128,
  "max_position_embeddings": 128,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000,
  "tie_word_embeddings": true
})";
    expect(static_cast<bool>(cfg), "write config.json");
  }

  constexpr std::int64_t hidden = 32;
  constexpr std::int64_t heads = 4;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t head_dim = hidden / heads;
  constexpr std::int64_t q_dim = heads * head_dim;
  constexpr std::int64_t kv_dim = kv_heads * head_dim;
  constexpr std::int64_t inter = 64;
  constexpr std::int64_t vocab = 128;

  std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>> tensors;
  put_tensor(&tensors, "model.embed_tokens.weight", {vocab, hidden}, 1.f);
  put_tensor(&tensors, "model.norm.weight", {hidden}, 2.f);

  float seed = 10.f;
  for (int li = 0; li < 2; ++li) {
    const std::string pfx = "model.layers." + std::to_string(li) + ".";
    put_tensor(&tensors, pfx + "input_layernorm.weight", {hidden}, seed++);
    put_tensor(&tensors, pfx + "self_attn.q_proj.weight", {q_dim, hidden}, seed++);
    put_tensor(&tensors, pfx + "self_attn.k_proj.weight", {kv_dim, hidden}, seed++);
    put_tensor(&tensors, pfx + "self_attn.v_proj.weight", {kv_dim, hidden}, seed++);
    put_tensor(&tensors, pfx + "self_attn.o_proj.weight", {hidden, q_dim}, seed++);
    put_tensor(&tensors, pfx + "post_attention_layernorm.weight", {hidden}, seed++);
    put_tensor(&tensors, pfx + "mlp.gate_proj.weight", {inter, hidden}, seed++);
    put_tensor(&tensors, pfx + "mlp.up_proj.weight", {inter, hidden}, seed++);
    put_tensor(&tensors, pfx + "mlp.down_proj.weight", {hidden, inter}, seed++);
  }

  const fs::path st_path = dir / "model.safetensors";
  const uaii::Error werr =
      uaii::loaders::safetensors_write_f32(st_path.string(), tensors, {{"format", "uaii-test"}});
  expect(werr.ok(), "safetensors_write_f32");
  if (!werr.ok()) {
    fs::remove_all(dir, ec);
    return;
  }

  uaii::loaders::HfLoader loader;
  expect(loader.accepts(dir.string()), "HfLoader::accepts");
  uaii::ir::Graph graph;
  const uaii::Error lerr = loader.load(dir.string(), &graph);
  expect(lerr.ok(), "HfLoader::load");
  if (lerr.ok()) {
    auto it = graph.metadata.find("n_layers");
    expect(it != graph.metadata.end() && it->second == "2", "metadata n_layers=2");
    int rope_nodes = 0;
    for (const auto& n : graph.nodes) {
      if (n.op_name == "RoPE") ++rope_nodes;
    }
    expect(rope_nodes >= 4, "RoPE nodes present (Q/K per layer)");
  }

  fs::remove_all(dir, ec);
}

void test_hf_gpt2_layout() {
  const fs::path dir = fs::temp_directory_path() / "uaii_hf_gpt2_test_tiny";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  {
    std::ofstream cfg(dir / "config.json");
    cfg << R"({
  "architectures": ["GPT2LMHeadModel"],
  "model_type": "gpt2",
  "n_embd": 32,
  "n_layer": 1,
  "n_head": 4,
  "vocab_size": 64,
  "n_positions": 64,
  "layer_norm_epsilon": 1e-5,
  "activation_function": "gelu_new"
})";
  }

  constexpr std::int64_t hidden = 32;
  constexpr std::int64_t vocab = 64;
  std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>> tensors;
  put_tensor(&tensors, "transformer.wte.weight", {vocab, hidden}, 1.f);
  put_tensor(&tensors, "transformer.ln_f.weight", {hidden}, 2.f);
  put_tensor(&tensors, "transformer.ln_f.bias", {hidden}, 2.5f);
  put_tensor(&tensors, "transformer.h.0.ln_1.weight", {hidden}, 3.f);
  put_tensor(&tensors, "transformer.h.0.ln_1.bias", {hidden}, 3.5f);
  put_tensor(&tensors, "transformer.h.0.attn.c_attn.weight", {hidden, 3 * hidden}, 4.f);
  put_tensor(&tensors, "transformer.h.0.attn.c_attn.bias", {3 * hidden}, 4.5f);
  put_tensor(&tensors, "transformer.h.0.attn.c_proj.weight", {hidden, hidden}, 5.f);
  put_tensor(&tensors, "transformer.h.0.ln_2.weight", {hidden}, 6.f);
  put_tensor(&tensors, "transformer.h.0.ln_2.bias", {hidden}, 6.5f);
  put_tensor(&tensors, "transformer.h.0.mlp.c_fc.weight", {hidden, hidden * 4}, 7.f);
  put_tensor(&tensors, "transformer.h.0.mlp.c_proj.weight", {hidden * 4, hidden}, 8.f);

  const uaii::Error werr =
      uaii::loaders::safetensors_write_f32((dir / "model.safetensors").string(), tensors, {});
  expect(werr.ok(), "gpt2 safetensors write");
  uaii::loaders::HfLoader loader;
  uaii::ir::Graph graph;
  const uaii::Error lerr = loader.load(dir.string(), &graph);
  expect(lerr.ok(), "HfLoader GPT-2 layout");
  if (!lerr.ok()) {
    std::cerr << "  gpt2 load: " << lerr.to_string() << '\n';
  } else {
    bool slice = false, gelu = false, ln = false;
    for (const auto& n : graph.nodes) {
      if (n.op_name == "Slice") slice = true;
      if (n.op_name == "Gelu") gelu = true;
      if (n.op_name == "LayerNorm") ln = true;
    }
    expect(slice, "GPT-2 fused QKV slices");
    expect(gelu, "GPT-2 GELU mlp");
    expect(ln, "GPT-2 LayerNorm");
  }
  fs::remove_all(dir, ec);
}

void test_parse_llama3_rope_scaling() {
  const char* json = R"({
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000,
    "rope_scaling": {
      "rope_type": "llama3",
      "factor": 8.0,
      "low_freq_factor": 1.0,
      "high_freq_factor": 4.0,
      "original_max_position_embeddings": 8192
    }
  })";
  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(err.ok(), "llama3 rope_scaling parses");
  expect(cfg.rope_scaling_type == "llama3", "rope_scaling_type llama3");
  expect(cfg.rope_scale_factor == 8.f, "rope factor 8");
  expect(cfg.rope_low_freq_factor == 1.f, "low_freq_factor");
  expect(cfg.rope_high_freq_factor == 4.f, "high_freq_factor");
  expect(cfg.rope_original_max_pos == 8192.f, "original_max_position_embeddings");
}

void test_reject_unknown_rope_scaling() {
  const char* json = R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rope_scaling": {"type": "longrope", "factor": 4}
  })";
  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(!err.ok(), "reject unknown rope_scaling.type");
  expect(err.message().find("unsupported rope_scaling.type=longrope") != std::string::npos,
         "names rope_scaling.type");
}

void test_parse_gpt_oss() {
  const char* json = R"({
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "quantization_config": {"quant_method": "mxfp4"}
  })";
  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(err.ok(), "gpt_oss config parses");
  expect(cfg.model_type == "gpt_oss", "model_type gpt_oss");
  expect(cfg.mxfp4, "mxfp4 flag");
  expect(uaii::loaders::hf_family_name(cfg.architecture, cfg.model_type) == "gpt_oss",
         "family gpt_oss");
}

void test_index_unique_shards() {
  const char* json = R"({
    "weight_map": {
      "a.weight": "model-00001-of-00002.safetensors",
      "b.weight": "model-00002-of-00002.safetensors",
      "c.weight": "model-00001-of-00002.safetensors"
    }
  })";
  std::vector<std::string> shards;
  const uaii::Error err = uaii::loaders::hf_index_unique_shards(json, &shards);
  expect(err.ok(), "hf_index_unique_shards ok");
  expect(shards.size() == 2, "two unique shards");
}

void write_f8_fixture(const fs::path& path) {
  const std::string header =
      R"({"t":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}})";
  const std::uint64_t n = static_cast<std::uint64_t>(header.size());
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  const char zeros[4] = {};
  out.write(zeros, 4);
}

void test_safetensors_dtypes() {
  const fs::path dir = fs::temp_directory_path() / "uaii_p0_dtype";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>> tensors;
  uaii::Shape sh;
  sh.dims = {8};
  std::vector<float> gold(8);
  for (int i = 0; i < 8; ++i) gold[static_cast<std::size_t>(i)] = 0.5f * static_cast<float>(i) - 1.f;
  tensors["w"] = {sh, gold};

  for (const char* dt : {"F32", "F16", "BF16"}) {
    const fs::path p = dir / (std::string(dt) + ".safetensors");
    const uaii::Error werr = uaii::loaders::safetensors_write(p.string(), tensors, dt, {});
    expect(werr.ok(), "write dtype");
    uaii::loaders::SafetensorsFile sf;
    const uaii::Error herr = uaii::loaders::safetensors_read_header(p.string(), &sf);
    expect(herr.ok(), "read header");
    std::vector<float> got;
    uaii::Shape out_sh;
    const uaii::Error lerr = uaii::loaders::safetensors_load_tensor_f32(sf, "w", &got, &out_sh);
    expect(lerr.ok(), "load f32 from stored dtype");
    expect(got.size() == gold.size(), "numel");
    float max_err = 0.f;
    for (std::size_t i = 0; i < gold.size(); ++i) {
      max_err = std::max(max_err, std::fabs(got[i] - gold[i]));
    }
    const float tol = (std::string(dt) == "F32") ? 1e-6f : (std::string(dt) == "F16" ? 2e-3f : 1e-2f);
    expect(max_err < tol, "round-trip error within dtype");
  }

  const fs::path f8 = dir / "f8.safetensors";
  write_f8_fixture(f8);
  uaii::loaders::SafetensorsFile sf;
  expect(uaii::loaders::safetensors_read_header(f8.string(), &sf).ok(), "f8 header");
  std::vector<float> got;
  const uaii::Error ferr = uaii::loaders::safetensors_load_tensor_f32(sf, "t", &got, nullptr);
  expect(!ferr.ok(), "F8 rejected");
  expect(ferr.message().find("Convert this checkpoint to BF16") != std::string::npos,
         "F8 convert hint");

  fs::remove_all(dir, ec);
}

void write_llama_cfg(const fs::path& dir, const char* extra = "") {
  std::ofstream cfg(dir / "config.json");
  cfg << R"({
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 32,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "intermediate_size": 64,
  "vocab_size": 32,
  "max_position_embeddings": 128,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000,
  "tie_word_embeddings": true
)" << extra << "}";
}

std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>> tiny_llama_weights() {
  constexpr std::int64_t hidden = 32;
  constexpr std::int64_t q_dim = 32;
  constexpr std::int64_t kv_dim = 16;
  constexpr std::int64_t inter = 64;
  constexpr std::int64_t vocab = 32;
  std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>> tensors;
  put_tensor(&tensors, "model.embed_tokens.weight", {vocab, hidden}, 1.f);
  put_tensor(&tensors, "model.norm.weight", {hidden}, 2.f);
  put_tensor(&tensors, "model.layers.0.input_layernorm.weight", {hidden}, 3.f);
  put_tensor(&tensors, "model.layers.0.self_attn.q_proj.weight", {q_dim, hidden}, 4.f);
  put_tensor(&tensors, "model.layers.0.self_attn.k_proj.weight", {kv_dim, hidden}, 5.f);
  put_tensor(&tensors, "model.layers.0.self_attn.v_proj.weight", {kv_dim, hidden}, 6.f);
  put_tensor(&tensors, "model.layers.0.self_attn.o_proj.weight", {hidden, q_dim}, 7.f);
  put_tensor(&tensors, "model.layers.0.post_attention_layernorm.weight", {hidden}, 8.f);
  put_tensor(&tensors, "model.layers.0.mlp.gate_proj.weight", {inter, hidden}, 9.f);
  put_tensor(&tensors, "model.layers.0.mlp.up_proj.weight", {inter, hidden}, 10.f);
  put_tensor(&tensors, "model.layers.0.mlp.down_proj.weight", {hidden, inter}, 11.f);
  return tensors;
}

void test_bf16_llama_load_and_inspect() {
  const fs::path dir = fs::temp_directory_path() / "uaii_p0_bf16_llama";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  write_llama_cfg(dir);
  auto tensors = tiny_llama_weights();
  const uaii::Error werr =
      uaii::loaders::safetensors_write((dir / "model.safetensors").string(), tensors, "BF16", {});
  expect(werr.ok(), "write BF16 llama");

  uaii::loaders::HfLoader loader;
  uaii::ir::Graph graph;
  const uaii::Error lerr = loader.load(dir.string(), &graph);
  expect(lerr.ok(), "load BF16 llama folder");
  if (!lerr.ok()) std::cerr << "  " << lerr.to_string() << '\n';

  uaii::loaders::HfCapability cap;
  expect(uaii::loaders::hf_inspect_directory(dir.string(), &cap).ok(), "inspect bf16");
  expect(cap.family == "llama", "family llama");
  expect(cap.weight_dtype.find("BF16") != std::string::npos, "weights: BF16");
  expect(cap.ok(), "BF16 llama inspect ok");
  const std::string report = uaii::loaders::hf_format_capability(cap);
  expect(report.find("family=llama") != std::string::npos, "report family");
  expect(report.find("blocked:") == std::string::npos, "no blocked");

  fs::remove_all(dir, ec);
}

void test_inspect_gptq_mla_gptoss() {
  std::error_code ec;

  {
    const fs::path dir = fs::temp_directory_path() / "uaii_p0_gptq";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    write_llama_cfg(dir, R"(  , "quantization_config": {"quant_method": "gptq"}
)");
    auto tensors = tiny_llama_weights();
    put_tensor(&tensors, "model.layers.0.self_attn.q_proj.qweight", {32, 8}, 1.f);
    expect(uaii::loaders::safetensors_write_f32((dir / "model.safetensors").string(), tensors, {})
               .ok(),
           "gptq weights");
    uaii::loaders::HfCapability cap;
    expect(uaii::loaders::hf_inspect_directory(dir.string(), &cap).ok(), "inspect gptq");
    bool blocked_q = false;
    for (const auto& b : cap.blocked) {
      if (b.find("quantized safetensors") != std::string::npos) blocked_q = true;
    }
    expect(blocked_q, "blocked: quantized safetensors");
    uaii::ir::Graph g;
    const uaii::Error lerr = uaii::loaders::HfLoader{}.load(dir.string(), &g);
    expect(!lerr.ok(), "gptq load fails");
    expect(lerr.message().find("quantized safetensors") != std::string::npos, "gptq load message");
    fs::remove_all(dir, ec);
  }

  {
    const fs::path dir = fs::temp_directory_path() / "uaii_p0_mla";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    {
      std::ofstream cfg(dir / "config.json");
      cfg << R"({
  "architectures": ["DeepseekV3ForCausalLM"],
  "model_type": "deepseek_v3",
  "hidden_size": 32,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "intermediate_size": 64,
  "vocab_size": 32,
  "max_position_embeddings": 128
})";
    }
    auto tensors = tiny_llama_weights();
    put_tensor(&tensors, "model.layers.0.self_attn.q_a_proj.weight", {8, 32}, 1.f);
    put_tensor(&tensors, "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", {8, 32}, 2.f);
    expect(uaii::loaders::safetensors_write_f32((dir / "model.safetensors").string(), tensors, {})
               .ok(),
           "mla weights");
    uaii::loaders::HfCapability cap;
    expect(uaii::loaders::hf_inspect_directory(dir.string(), &cap).ok(), "inspect mla");
    expect(cap.family == "kimi_mla", "family kimi_mla");
    expect(cap.mla, "mla flag");
    bool blocked_mla = false;
    for (const auto& b : cap.blocked) {
      if (b.find("HF MLA; use GGUF") != std::string::npos) blocked_mla = true;
    }
    expect(blocked_mla, "blocked: HF MLA; use GGUF");
    uaii::ir::Graph g;
    const uaii::Error lerr = uaii::loaders::HfLoader{}.load(dir.string(), &g);
    expect(!lerr.ok() && lerr.message() == "HF MLA; use GGUF", "MLA load matches inspect");
    fs::remove_all(dir, ec);
  }

  {
    const fs::path dir = fs::temp_directory_path() / "uaii_p0_gptoss";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    {
      std::ofstream cfg(dir / "config.json");
      cfg << R"({
  "architectures": ["GptOssForCausalLM"],
  "model_type": "gpt_oss",
  "hidden_size": 32,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "intermediate_size": 64,
  "vocab_size": 32,
  "max_position_embeddings": 128,
  "quantization_config": {"quant_method": "mxfp4"}
})";
    }
    auto tensors = tiny_llama_weights();
    expect(uaii::loaders::safetensors_write_f32((dir / "model.safetensors").string(), tensors, {})
               .ok(),
           "gptoss weights");
    uaii::loaders::HfCapability cap;
    expect(uaii::loaders::hf_inspect_directory(dir.string(), &cap).ok(), "inspect gpt_oss");
    expect(cap.family == "gpt_oss", "family=gpt_oss");
    bool blocked_mx = false;
    for (const auto& b : cap.blocked) {
      if (b.find("mxfp4 convert to BF16/GGUF") != std::string::npos) blocked_mx = true;
    }
    expect(blocked_mx, "blocked mxfp4");
    uaii::ir::Graph g;
    const uaii::Error lerr = uaii::loaders::HfLoader{}.load(dir.string(), &g);
    expect(!lerr.ok(), "gpt-oss mxfp4 load fail");
    expect(lerr.message().find("mxfp4 convert to BF16/GGUF") != std::string::npos,
           "mxfp4 load message");
    fs::remove_all(dir, ec);
  }

  {
    const fs::path dir = fs::temp_directory_path() / "uaii_p0_gptoss_bf16";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    {
      std::ofstream cfg(dir / "config.json");
      cfg << R"({
  "architectures": ["GptOssForCausalLM"],
  "model_type": "gpt_oss",
  "hidden_size": 32,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "intermediate_size": 64,
  "vocab_size": 32,
  "max_position_embeddings": 128
})";
    }
    auto tensors = tiny_llama_weights();
    expect(uaii::loaders::safetensors_write((dir / "model.safetensors").string(), tensors, "BF16",
                                           {})
               .ok(),
           "gptoss bf16 weights");
    uaii::loaders::HfCapability cap;
    expect(uaii::loaders::hf_inspect_directory(dir.string(), &cap).ok(), "inspect gpt_oss dense");
    expect(cap.family == "gpt_oss", "family=gpt_oss dense");
    bool blocked_h = false;
    for (const auto& b : cap.blocked) {
      if (b.find("gpt-oss family handler") != std::string::npos) blocked_h = true;
    }
    expect(blocked_h, "dense gpt-oss not Llama-wired");
    uaii::ir::Graph g;
    expect(!uaii::loaders::HfLoader{}.load(dir.string(), &g).ok(), "dense gpt-oss load fail-closed");
    fs::remove_all(dir, ec);
  }
}

void test_gguf_mla_inspect() {
  const fs::path path = fs::temp_directory_path() / "uaii_p0_kimi.gguf";
  std::error_code ec;
  fs::remove(path, ec);

  std::unordered_map<std::string, uaii::loaders::GgufValue> kv;
  kv["general.architecture"] = std::string{"deepseekv3"};
  std::vector<std::pair<uaii::loaders::GgufTensorInfo, std::vector<float>>> tensors;
  auto add = [&](const std::string& name, std::vector<std::uint64_t> dims) {
    uaii::loaders::GgufTensorInfo ti;
    ti.name = name;
    ti.dims = std::move(dims);
    std::size_t n = 1;
    for (auto d : ti.dims) n *= static_cast<std::size_t>(d);
    tensors.emplace_back(std::move(ti), std::vector<float>(n, 0.1f));
  };
  add("token_embd.weight", {8, 4});
  add("blk.0.attn_kv_a_mqa.weight", {8, 4});
  add("blk.0.attn_k_b.weight", {8, 4});
  add("blk.0.attn_v_b.weight", {8, 4});
  add("blk.0.ffn_gate_inp.weight", {4, 4});
  const uaii::Error werr = uaii::loaders::gguf_write_f32(path.string(), kv, tensors);
  expect(werr.ok(), "write kimi gguf fixture");
  uaii::loaders::GgufCapability cap;
  expect(uaii::loaders::gguf_inspect_file(path.string(), &cap).ok(), "gguf inspect");
  expect(cap.mla, "gguf mla");
  expect(cap.family == "kimi_mla", "gguf family kimi_mla");
  expect(cap.ok(), "gguf mla path ok");
  const std::string report = uaii::loaders::gguf_format_capability(cap);
  expect(report.find("family=kimi_mla") != std::string::npos, "gguf report family");
  fs::remove(path, ec);
}

}  // namespace

int main() {
  test_parse_config();
  test_reject_unknown_arch();
  test_accept_generic_causal();
  test_moe_config_ok();
  test_architecture_supported();
  test_hf_loader_roundtrip();
  test_hf_gpt2_layout();
  test_parse_llama3_rope_scaling();
  test_reject_unknown_rope_scaling();
  test_parse_gpt_oss();
  test_index_unique_shards();
  test_safetensors_dtypes();
  test_bf16_llama_load_and_inspect();
  test_inspect_gptq_mla_gptoss();
  test_gguf_mla_inspect();

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_hf_loader_test: OK\n";
  return EXIT_SUCCESS;
}
