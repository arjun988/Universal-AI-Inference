#include "uaii/ir/graph.hpp"
#include "uaii/loaders/hf.hpp"
#include "uaii/loaders/safetensors.hpp"

#include <cmath>
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
  expect(!err.ok(), "reject unknown architecture");
}

void test_reject_moe() {
  const char* json = R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "num_local_experts": 8
  })";
  uaii::loaders::HfConfig cfg;
  const uaii::Error err = uaii::loaders::hf_parse_config_json(json, &cfg);
  expect(!err.ok(), "reject MoE config");
}

void test_architecture_supported() {
  expect(uaii::loaders::hf_architecture_supported("LlamaForCausalLM"), "llama supported");
  expect(uaii::loaders::hf_architecture_supported("Qwen2ForCausalLM"), "qwen2 supported");
  expect(uaii::loaders::hf_architecture_supported("Gemma2ForCausalLM"), "gemma2 supported");
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

}  // namespace

int main() {
  test_parse_config();
  test_reject_unknown_arch();
  test_reject_moe();
  test_architecture_supported();
  test_hf_loader_roundtrip();

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_hf_loader_test: OK\n";
  return EXIT_SUCCESS;
}
