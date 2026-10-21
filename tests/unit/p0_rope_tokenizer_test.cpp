#include "uaii/ir/graph.hpp"
#include "uaii/kernels/kernels.hpp"
#include "uaii/loaders/gguf.hpp"
#include "uaii/loaders/hf.hpp"
#include "uaii/loaders/registry.hpp"
#include "uaii/loaders/safetensors.hpp"
#include "uaii/runtime/session.hpp"
#include "uaii/tokenizers/bpe_tokenizer.hpp"
#include "uaii/tokenizers/chat_template.hpp"
#include "uaii/tokenizers/hf_tokenizer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

void expect_close(float a, float b, float tol, const char* msg) {
  if (std::fabs(a - b) > tol) {
    std::cerr << "FAIL: " << msg << " got " << a << " expected " << b << '\n';
    ++failures;
  }
}

void test_llama3_rope_freqs() {
  constexpr std::int64_t n_pairs = 64;
  constexpr float theta = 500000.f;
  constexpr float factor = 8.f;
  constexpr float old_ctx = 8192.f;

  const float u0 = uaii::kernels::rope_scaled_inv_freq(0, n_pairs, theta, "", 1.f, 1.f, 4.f, old_ctx);
  expect_close(u0, 1.f, 1e-6f, "unscaled pair0 is 1");

  const float s0 = uaii::kernels::rope_scaled_inv_freq(0, n_pairs, theta, "llama3", factor, 1.f, 4.f,
                                                       old_ctx);
  expect_close(s0, u0, 1e-6f, "llama3 high-freq pair unchanged");

  const float u63 =
      uaii::kernels::rope_scaled_inv_freq(63, n_pairs, theta, "", 1.f, 1.f, 4.f, old_ctx);
  const float s63 = uaii::kernels::rope_scaled_inv_freq(63, n_pairs, theta, "llama3", factor, 1.f,
                                                        4.f, old_ctx);
  expect_close(s63, u63 / factor, 1e-5f, "llama3 low-freq pair / factor");

  const float linear =
      uaii::kernels::rope_scaled_inv_freq(3, n_pairs, theta, "linear", 2.f, 1.f, 4.f, old_ctx);
  const float base =
      uaii::kernels::rope_scaled_inv_freq(3, n_pairs, theta, "", 1.f, 1.f, 4.f, old_ctx);
  expect_close(linear, base / 2.f, 1e-6f, "linear scaling");

  for (int i = 0; i < 8; ++i) {
    const float got = uaii::kernels::rope_scaled_inv_freq(i, 64, 10000.f, "", 1.f, 1.f, 4.f, 8192.f);
    const float ref = 1.f / std::pow(10000.f, (2.f * static_cast<float>(i)) / 128.f);
    expect_close(got, ref, 2e-5f, "unscaled first-8 closed form");
  }

  // Independent Python/Meta Llama3ScalingRotaryEmbedding values (Llama-3.1 8B).
  const float py_llama3[][3] = {
      {0, 1.000000000e+00f, 1.000000000e+00f},
      {7, 2.380538205e-01f, 2.380538205e-01f},
      {32, 1.414213562e-03f, 5.248461610e-04f},
      {40, 2.742481757e-04f, 3.428102196e-05f},
      {63, 2.455140791e-06f, 3.068925989e-07f},
  };
  for (const auto& row : py_llama3) {
    const auto i = static_cast<std::int64_t>(row[0]);
    expect_close(uaii::kernels::rope_scaled_inv_freq(i, 64, theta, "", 1.f, 1.f, 4.f, old_ctx),
                 row[1], 2e-6f, "unscaled vs python");
    expect_close(uaii::kernels::rope_scaled_inv_freq(i, 64, theta, "llama3", factor, 1.f, 4.f, old_ctx),
                 row[2], 2e-6f, "llama3 vs python");
  }
}

void test_partial_rotary_leading_and_trailing() {
  float in[4] = {1.f, 0.f, 0.f, 1.f};
  float pos[1] = {1.f};
  float leading_out[4] = {};
  float trailing_out[4] = {};
  const std::int64_t shape[] = {1, 4};
  const std::int64_t pshape[] = {1};
  uaii::kernels::TensorView vin;
  vin.dtype = uaii::DType::F32;
  vin.shape = shape;
  vin.rank = 2;
  vin.data = in;
  vin.nbytes = sizeof(in);
  uaii::kernels::TensorView vpos = vin;
  vpos.shape = pshape;
  vpos.rank = 1;
  vpos.data = pos;
  vpos.nbytes = sizeof(pos);
  uaii::kernels::TensorView vout = vin;
  vout.data = leading_out;
  expect(uaii::kernels::rope_f32(vin, &vpos, &vout, 10000.f, 4, 0, 2, nullptr, 1.f, 1.f, 4.f,
                                 8192.f, 1.f, nullptr, nullptr, 0, /*rotary_align=*/0)
             .ok(),
         "leading partial rope");
  expect_close(leading_out[0], std::cos(1.f), 1e-5f, "phi leading cos");
  expect_close(leading_out[1], std::sin(1.f), 1e-5f, "phi leading sin");
  expect_close(leading_out[2], 0.f, 1e-5f, "phi tail dim2 copied");
  expect_close(leading_out[3], 1.f, 1e-5f, "phi tail dim3 copied");

  vout.data = trailing_out;
  expect(uaii::kernels::rope_f32(vin, &vpos, &vout, 10000.f, 4, 0, 2, nullptr, 1.f, 1.f, 4.f,
                                 8192.f, 1.f, nullptr, nullptr, 0, /*rotary_align=*/1)
             .ok(),
         "trailing partial rope");
  expect_close(trailing_out[0], 1.f, 1e-5f, "mla prefix dim0 copied");
  expect_close(trailing_out[1], 0.f, 1e-5f, "mla prefix dim1 copied");
  expect_close(trailing_out[2], -std::sin(1.f), 1e-5f, "mla tail cos term");
  expect_close(trailing_out[3], std::cos(1.f), 1e-5f, "mla tail sin term");
}

void test_yarn_ramp_matches_hf() {
  // dim=8 (4 pairs), theta=10000, factor=4, original context 2048.
  // HF correction range: low=floor(corr(beta_fast=32))=1, high=ceil(corr(beta_slow=1))=3.
  const float y0 =
      uaii::kernels::rope_scaled_inv_freq(0, 4, 10000.f, "yarn", 4.f, 1.f, 4.f, 2048.f);
  expect_close(y0, 1.f, 1e-5f, "yarn high-freq is extrapolation");
  const float y3 =
      uaii::kernels::rope_scaled_inv_freq(3, 4, 10000.f, "yarn", 4.f, 1.f, 4.f, 2048.f);
  expect_close(y3, 0.00025f, 1e-7f, "yarn low-freq is interpolation");
  const float y2 =
      uaii::kernels::rope_scaled_inv_freq(2, 4, 10000.f, "yarn", 4.f, 1.f, 4.f, 2048.f);
  expect_close(y2, 0.00625f, 1e-6f, "yarn mid-band blends");
}

void test_rope_kernel_applies_scaling() {
  float in[8] = {1, 0, 0, 1, 0.5f, 0.5f, 0, 0};
  float out[8] = {};
  const std::int64_t shape[] = {1, 8};
  uaii::kernels::TensorView vin;
  vin.dtype = uaii::DType::F32;
  vin.shape = shape;
  vin.rank = 2;
  vin.data = in;
  vin.nbytes = sizeof(in);
  uaii::kernels::TensorView vout = vin;
  vout.data = out;
  const uaii::Error err =
      uaii::kernels::rope_f32(vin, nullptr, &vout, 10000.f, 8, 0, 0, "linear", 2.f);
  expect(err.ok(), "rope_f32 linear");
  // Position 0 → identity rotation regardless of frequency.
  expect_close(out[0], in[0], 1e-5f, "pos0 identity x0");
  expect_close(out[1], in[1], 1e-5f, "pos0 identity x1");
}

void test_rmsnorm_offset_and_qk() {
  float x[4] = {3.f, 4.f, 0.f, 0.f};
  float w[4] = {1.f, 1.f, 1.f, 1.f};
  float out0[4] = {};
  float out1[4] = {};
  const std::int64_t sh2[] = {1, 4};
  uaii::kernels::TensorView vin;
  vin.dtype = uaii::DType::F32;
  vin.shape = sh2;
  vin.rank = 2;
  vin.data = x;
  vin.nbytes = sizeof(x);
  uaii::kernels::TensorView vw = vin;
  const std::int64_t shw[] = {4};
  vw.shape = shw;
  vw.rank = 1;
  vw.data = w;
  vw.nbytes = sizeof(w);
  uaii::kernels::TensorView v0 = vin;
  v0.data = out0;
  uaii::kernels::TensorView v1 = vin;
  v1.data = out1;
  expect(uaii::kernels::rmsnorm_f32(vin, &vw, &v0, 1e-6f, 0.f).ok(), "rms offset 0");
  expect(uaii::kernels::rmsnorm_f32(vin, &vw, &v1, 1e-6f, 1.f).ok(), "rms offset 1");
  // ms = (9+16)/4 = 6.25, inv = 1/sqrt(6.25)=0.4; offset0: 3*0.4*1=1.2; offset1: 3*0.4*2=2.4
  expect_close(out0[0], 1.2f, 1e-5f, "llama-style *w");
  expect_close(out1[0], 2.4f, 1e-5f, "gemma-style *(1+w)");

  float qx[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  float qw[4] = {2, 2, 2, 2};
  float qout[8] = {};
  const std::int64_t qsh[] = {1, 8};
  const std::int64_t qwsh[] = {4};
  uaii::kernels::TensorView qin = vin;
  qin.shape = qsh;
  qin.data = qx;
  qin.nbytes = sizeof(qx);
  uaii::kernels::TensorView qwv = vw;
  qwv.shape = qwsh;
  qwv.rank = 1;
  qwv.data = qw;
  qwv.nbytes = sizeof(qw);
  uaii::kernels::TensorView qvo = qin;
  qvo.data = qout;
  expect(uaii::kernels::rmsnorm_f32(qin, &qwv, &qvo, 1e-6f, 0.f).ok(), "per-head qk-norm");
  // Head 0 is [1,0,0,0]: RMS = sqrt(1/4)=0.5, inv=2, *w=2 → 4.
  expect_close(qout[0], 4.f, 1e-4f, "head0 first after qk-norm");
}

void test_sliding_window_mask() {
  // Two query positions, matching KV seq. Window=1 masks token 1 from key 0.
  float q[4] = {1.f, 0.f, 1.f, 0.f};
  float k[4] = {1.f, 0.f, 0.f, 1.f};
  float v[4] = {1.f, 0.f, 0.f, 1.f};
  float out[4] = {};
  const std::int64_t sh[] = {1, 2, 2};
  uaii::kernels::TensorView vq, vk, vv, vo;
  vq.dtype = vk.dtype = vv.dtype = vo.dtype = uaii::DType::F32;
  vq.shape = vk.shape = vv.shape = vo.shape = sh;
  vq.rank = vk.rank = vv.rank = vo.rank = 3;
  vq.data = q;
  vq.nbytes = sizeof(q);
  vk.data = k;
  vk.nbytes = sizeof(k);
  vv.data = v;
  vv.nbytes = sizeof(v);
  vo.data = out;
  vo.nbytes = sizeof(out);
  expect(uaii::kernels::attention_f32(vq, vk, vv, &vo, 1, 1.f, true, 0.f, 1).ok(),
         "attn sliding=1");
  // Token 0 attends only to key 0 → V[0] = (1, 0). Token 1 attends only to key 1 → V[1] = (0, 1).
  expect_close(out[0], 1.f, 1e-4f, "slide token0 dim0");
  expect_close(out[1], 0.f, 1e-4f, "slide token0 dim1");
  expect_close(out[2], 0.f, 1e-4f, "slide token1 dim0");
  expect_close(out[3], 1.f, 1e-4f, "slide token1 dim1");
}

std::string utf8_encode(char32_t cp) {
  std::string out;
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

std::vector<std::string> gpt2_byte_tokens() {
  std::vector<int> bs;
  for (int i = 33; i <= 126; ++i) bs.push_back(i);
  for (int i = 161; i <= 172; ++i) bs.push_back(i);
  for (int i = 174; i <= 255; ++i) bs.push_back(i);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    bool found = false;
    for (int x : bs) {
      if (x == b) {
        found = true;
        break;
      }
    }
    if (!found) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  std::vector<std::string> tokens(256);
  for (std::size_t i = 0; i < bs.size(); ++i) {
    tokens[static_cast<std::size_t>(bs[i])] = utf8_encode(static_cast<char32_t>(cs[i]));
  }
  return tokens;
}

void test_byte_level_bpe_roundtrip() {
  auto tokens = gpt2_byte_tokens();
  tokens.push_back("<|eot_id|>");
  uaii::tokenizers::BpeTokenizer tok;
  expect(tok.load_from_vocab_merges(tokens, {}).ok(), "load byte vocab");
  tok.set_byte_level_pretokenize(true);
  tok.register_special("<|eot_id|>");

  const std::string samples[] = {"hello", "你好", "hi 👋", "a\nb"};
  for (const auto& s : samples) {
    std::vector<std::int64_t> ids;
    expect(tok.encode(s, &ids).ok(), "encode");
    expect(!ids.empty(), "non-empty ids");
    std::string back;
    expect(tok.decode(ids, &back).ok(), "decode");
    expect(back == s, "round-trip");
    if (back != s) {
      std::cerr << "  round-trip mismatch for [" << s << "] got [" << back << "]\n";
    }
  }

  std::vector<std::int64_t> ids;
  expect(tok.encode("hi<|eot_id|>", &ids).ok(), "encode special");
  expect(!ids.empty() && ids.back() == 256, "special is last id");
  std::string back;
  expect(tok.decode(ids, &back).ok(), "decode special");
  expect(back == "hi<|eot_id|>", "special round-trip");
}

void test_hf_tokenizer_json_pipeline() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "uaii_p0_tokjson";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  auto json_escape = [](const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
      if (c == '"') o += "\\\"";
      else if (c == '\\') o += "\\\\";
      else o.push_back(static_cast<char>(c));
    }
    return o;
  };

  auto tokens = gpt2_byte_tokens();
  tokens.push_back("<|begin_of_text|>");
  std::ostringstream vocab;
  vocab << "{";
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i) vocab << ",";
    vocab << "\"" << json_escape(tokens[i]) << "\":" << i;
  }
  vocab << "}";

  {
    std::ofstream out(dir / "tokenizer.json");
    out << "{\n  \"model\": { \"type\": \"BPE\", \"vocab\": " << vocab.str()
        << ", \"merges\": [] },\n"
        << "  \"pre_tokenizer\": { \"type\": \"ByteLevel\", \"add_prefix_space\": false },\n"
        << "  \"added_tokens\": [ { \"id\": 256, \"content\": \"<|begin_of_text|>\", "
           "\"special\": true } ]\n}\n";
  }

  uaii::Error err;
  auto tok = uaii::tokenizers::load_hf_tokenizer(dir.string(), &err);
  expect(err.ok() && tok, "load tokenizer.json");
  if (tok) {
    std::vector<std::int64_t> ids;
    expect(tok->encode("<|begin_of_text|>hi", &ids).ok(), "encode with special");
    expect(!ids.empty() && ids.front() == 256, "bos id first");
    std::string back;
    expect(tok->decode(ids, &back).ok(), "decode json tok");
    expect(back.find("<|begin_of_text|>") == 0, "bos preserved");
  }

  {
    std::ofstream out(dir / "tokenizer.json");
    out << R"({"model":{"type":"Unigram","vocab":{"a":0}}})";
  }
  uaii::Error uerr;
  auto utok = uaii::tokenizers::load_hf_tokenizer(dir.string(), &uerr);
  expect(!uerr.ok() && !utok, "Unigram rejected");
  expect(uerr.message().find("Unigram") != std::string::npos, "Unigram error names type");

  fs::remove_all(dir, ec);
}

std::vector<float> fill_f32(std::size_t n, float seed) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = std::sin(seed + static_cast<float>(i) * 0.017f) * 0.1f;
  }
  return v;
}

void put_f32(std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>>* tensors,
             const std::string& name, std::initializer_list<std::int64_t> dims, float seed) {
  uaii::Shape shape;
  shape.dims.assign(dims.begin(), dims.end());
  std::size_t n = 1;
  for (std::int64_t d : shape.dims) n *= static_cast<std::size_t>(d);
  (*tensors)[name] = {shape, fill_f32(n, seed)};
}

void test_bf16_generate_one_step() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "uaii_p0_bf16_gen";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  {
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
  "rope_scaling": {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192
  },
  "tie_word_embeddings": true
})";
  }
  std::unordered_map<std::string, std::pair<uaii::Shape, std::vector<float>>> tensors;
  put_f32(&tensors, "model.embed_tokens.weight", {32, 32}, 1.f);
  put_f32(&tensors, "model.norm.weight", {32}, 2.f);
  put_f32(&tensors, "model.layers.0.input_layernorm.weight", {32}, 3.f);
  put_f32(&tensors, "model.layers.0.self_attn.q_proj.weight", {32, 32}, 4.f);
  put_f32(&tensors, "model.layers.0.self_attn.k_proj.weight", {16, 32}, 5.f);
  put_f32(&tensors, "model.layers.0.self_attn.v_proj.weight", {16, 32}, 6.f);
  put_f32(&tensors, "model.layers.0.self_attn.o_proj.weight", {32, 32}, 7.f);
  put_f32(&tensors, "model.layers.0.post_attention_layernorm.weight", {32}, 8.f);
  put_f32(&tensors, "model.layers.0.mlp.gate_proj.weight", {64, 32}, 9.f);
  put_f32(&tensors, "model.layers.0.mlp.up_proj.weight", {64, 32}, 10.f);
  put_f32(&tensors, "model.layers.0.mlp.down_proj.weight", {32, 64}, 11.f);
  expect(uaii::loaders::safetensors_write((dir / "model.safetensors").string(), tensors, "BF16", {})
             .ok(),
         "write bf16 gen fixture");

  uaii::ir::Graph graph;
  const uaii::Error lerr = uaii::loaders::HfLoader{}.load(dir.string(), &graph);
  expect(lerr.ok(), "load bf16 llama for generate");
  if (!lerr.ok()) {
    std::cerr << "  " << lerr.to_string() << '\n';
    fs::remove_all(dir, ec);
    return;
  }
  bool rope_scaled = false;
  for (const auto& n : graph.nodes) {
    if (n.op_name != "RoPE") continue;
    for (const auto& a : n.attributes) {
      if (a.key == "scale_type" && std::holds_alternative<std::string>(a.value) &&
          std::get<std::string>(a.value) == "llama3") {
        rope_scaled = true;
      }
    }
  }
  expect(rope_scaled, "RoPE nodes carry llama3 scale_type");

  uaii::runtime::SessionOptions opts;
  opts.validate = true;
  opts.backend_name = "cpu";
  uaii::runtime::Session session;
  const uaii::Error cerr = session.create(std::move(graph), opts);
  expect(cerr.ok(), "session create bf16 llama");
  if (!cerr.ok()) {
    std::cerr << "  " << cerr.to_string() << '\n';
    fs::remove_all(dir, ec);
    return;
  }
  std::vector<std::int64_t> out;
  const uaii::Error gerr = session.generate({1, 2}, 1, &out);
  expect(gerr.ok(), "bf16 generate one step");
  if (!gerr.ok()) std::cerr << "  " << gerr.to_string() << '\n';
  expect(out.size() >= 3, "prompt plus at least one new token");
  fs::remove_all(dir, ec);
}

void add_gguf_f32(std::vector<std::pair<uaii::loaders::GgufTensorInfo, std::vector<float>>>* tensors,
                  const std::string& name, std::vector<std::uint64_t> dims, float seed) {
  uaii::loaders::GgufTensorInfo ti;
  ti.name = name;
  ti.dims = std::move(dims);
  std::size_t n = 1;
  for (auto d : ti.dims) n *= static_cast<std::size_t>(d);
  tensors->emplace_back(std::move(ti), fill_f32(n, seed));
}

void test_kimi_gguf_mla_generate() {
  namespace fs = std::filesystem;
  const fs::path path = fs::temp_directory_path() / "uaii_p0_kimi_mla.gguf";
  std::error_code ec;
  fs::remove(path, ec);

  // Tiny DeepSeek-V3 / Kimi MLA: 2 heads, nope=2, rope=2, kv_lora=4, dim=8, vocab=16.
  std::unordered_map<std::string, uaii::loaders::GgufValue> kv;
  kv["general.architecture"] = std::string{"deepseekv3"};
  kv["general.name"] = std::string{"uaii-tiny-kimi-mla"};
  kv["deepseekv3.block_count"] = std::int32_t{1};
  kv["deepseekv3.embedding_length"] = std::int32_t{8};
  kv["deepseekv3.attention.head_count"] = std::int32_t{2};
  kv["deepseekv3.attention.qk_nope_head_dim"] = std::int32_t{2};
  kv["deepseekv3.attention.qk_rope_head_dim"] = std::int32_t{2};
  kv["tokenizer.ggml.tokens"] = std::vector<std::string>{
      "<unk>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"};

  std::vector<std::pair<uaii::loaders::GgufTensorInfo, std::vector<float>>> tensors;
  add_gguf_f32(&tensors, "token_embd.weight", {8, 16}, 1.f);
  add_gguf_f32(&tensors, "output_norm.weight", {8}, 2.f);
  add_gguf_f32(&tensors, "blk.0.attn_norm.weight", {8}, 3.f);
  add_gguf_f32(&tensors, "blk.0.attn_q.weight", {8, 8}, 4.f);          // [out=8,in=8]
  add_gguf_f32(&tensors, "blk.0.attn_kv_a_mqa.weight", {8, 6}, 5.f);  // [out=6,in=8]
  add_gguf_f32(&tensors, "blk.0.attn_k_b.weight", {4, 4}, 6.f);        // [out=4,in=4] nope-only K
  add_gguf_f32(&tensors, "blk.0.attn_v_b.weight", {4, 8}, 7.f);        // [out=8,in=4]
  add_gguf_f32(&tensors, "blk.0.attn_output.weight", {8, 8}, 8.f);
  add_gguf_f32(&tensors, "blk.0.ffn_norm.weight", {8}, 9.f);
  add_gguf_f32(&tensors, "blk.0.ffn_gate.weight", {8, 16}, 10.f);
  add_gguf_f32(&tensors, "blk.0.ffn_up.weight", {8, 16}, 11.f);
  add_gguf_f32(&tensors, "blk.0.ffn_down.weight", {16, 8}, 12.f);

  const uaii::Error werr = uaii::loaders::gguf_write_f32(path.string(), kv, tensors);
  expect(werr.ok(), "write kimi mla gguf");

  uaii::loaders::GgufCapability cap;
  expect(uaii::loaders::gguf_inspect_file(path.string(), &cap).ok(), "inspect kimi gguf");
  expect(cap.family == "kimi_mla" && cap.mla && cap.ok(), "kimi gguf inspect ok");

  uaii::ir::Graph graph;
  const uaii::Error lerr = uaii::loaders::load_model(path.string(), &graph);
  expect(lerr.ok(), "load kimi mla gguf graph");
  if (!lerr.ok()) {
    std::cerr << "  " << lerr.to_string() << '\n';
    fs::remove(path, ec);
    return;
  }
  bool expand = false;
  for (const auto& n : graph.nodes) {
    if (n.op_name == "MLAExpandK") expand = true;
  }
  expect(expand, "MLAExpandK in graph");
  expect(graph.metadata.count("mla") && graph.metadata.at("mla") == "1", "graph meta mla=1");

  uaii::runtime::SessionOptions opts;
  opts.validate = true;
  opts.backend_name = "cpu";
  uaii::runtime::Session session;
  const uaii::Error cerr = session.create(std::move(graph), opts);
  expect(cerr.ok(), "session create kimi mla");
  if (!cerr.ok()) {
    std::cerr << "  " << cerr.to_string() << '\n';
    fs::remove(path, ec);
    return;
  }
  std::vector<std::int64_t> out;
  const uaii::Error gerr = session.generate({1}, 1, &out);
  expect(gerr.ok(), "kimi mla generate one token");
  if (!gerr.ok()) std::cerr << "  " << gerr.to_string() << '\n';
  expect(out.size() >= 2, "kimi mla emitted a token");
  fs::remove(path, ec);
}

void test_chat_multiturn_family() {
  using uaii::tokenizers::ChatMessage;
  using uaii::tokenizers::format_chat_messages;
  const std::vector<ChatMessage> turns = {
      {"system", "sys"},
      {"user", "u1"},
      {"assistant", "a1"},
      {"user", "u2"},
  };
  const std::string mis = format_chat_messages("mistral", turns);
  expect(mis.find("[INST] sys\n\nu1 [/INST]") != std::string::npos, "mistral first inst");
  expect(mis.find("a1</s>") != std::string::npos, "mistral assistant history");
  expect(mis.find("[INST] u2 [/INST]") != std::string::npos, "mistral second user");
  const std::string gem = format_chat_messages("gemma", turns);
  expect(gem.find("<start_of_turn>model\na1<end_of_turn>") != std::string::npos,
         "gemma keeps assistant history");
  expect(gem.find("sys\n\nu1") != std::string::npos, "gemma folds system into first user");
  const std::string l3 = format_chat_messages("llama3", turns);
  expect(l3.find("<|start_header_id|>assistant<|end_header_id|>\n\na1<|eot_id|>") !=
             std::string::npos,
         "llama3 assistant turn");
}

}  // namespace

int main() {
  test_llama3_rope_freqs();
  test_yarn_ramp_matches_hf();
  test_partial_rotary_leading_and_trailing();
  test_rope_kernel_applies_scaling();
  test_rmsnorm_offset_and_qk();
  test_sliding_window_mask();
  test_byte_level_bpe_roundtrip();
  test_hf_tokenizer_json_pipeline();
  test_bf16_generate_one_step();
  test_kimi_gguf_mla_generate();
  test_chat_multiturn_family();

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_p0_rope_tokenizer_test: OK\n";
  return EXIT_SUCCESS;
}
