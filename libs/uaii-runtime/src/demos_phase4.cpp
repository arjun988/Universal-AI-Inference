#include "uaii/runtime/session.hpp"

#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/loaders/gguf.hpp"
#include "uaii/loaders/registry.hpp"
#include "uaii/loaders/safetensors.hpp"
#include "uaii/tokenizers/simple_tokenizer.hpp"

#include <cmath>
#include <filesystem>
#include <numeric>
#include <unordered_map>

namespace uaii {
namespace runtime {
namespace {

bool nearly_one(float x, float eps = 1e-3f) {
  return std::fabs(x - 1.0f) <= eps;
}

std::string models_dir() {
  namespace fs = std::filesystem;
  if (fs::exists("examples/models")) {
    return "examples/models";
  }
  if (fs::exists("../examples/models")) {
    return "../examples/models";
  }
  fs::create_directories("uaii_phase4_models");
  return "uaii_phase4_models";
}

Error write_tiny_lm_gguf(const std::string& path, int vocab, int dim) {
  std::vector<float> emb(static_cast<std::size_t>(vocab * dim));
  std::vector<float> norm(static_cast<std::size_t>(dim), 1.0f);
  std::vector<float> outw(static_cast<std::size_t>(vocab * dim));
  for (int i = 0; i < vocab; ++i) {
    for (int d = 0; d < dim; ++d) {
      emb[static_cast<std::size_t>(i * dim + d)] = (i == d % vocab) ? 1.0f : 0.1f;
      outw[static_cast<std::size_t>(i * dim + d)] = (i == 3) ? 1.0f : 0.0f;  // bias to token 3
    }
  }

  loaders::GgufTensorInfo emb_i;
  emb_i.name = "token_embd.weight";
  emb_i.dims = {static_cast<std::uint64_t>(dim), static_cast<std::uint64_t>(vocab)};
  loaders::GgufTensorInfo norm_i;
  norm_i.name = "output_norm.weight";
  norm_i.dims = {static_cast<std::uint64_t>(dim)};
  loaders::GgufTensorInfo out_i;
  out_i.name = "output.weight";
  out_i.dims = {static_cast<std::uint64_t>(dim), static_cast<std::uint64_t>(vocab)};

  std::unordered_map<std::string, loaders::GgufValue> kv;
  kv["general.name"] = std::string("uaii-tiny-gguf");
  kv["general.architecture"] = std::string("tiny_lm");

  return loaders::gguf_write_f32(
      path, kv,
      {{emb_i, emb}, {norm_i, norm}, {out_i, outw}});
}

Error write_tiny_lm_safetensors(const std::string& path, int vocab, int dim) {
  std::vector<float> emb(static_cast<std::size_t>(vocab * dim), 0.1f);
  std::vector<float> norm(static_cast<std::size_t>(dim), 1.0f);
  std::vector<float> outw(static_cast<std::size_t>(vocab * dim), 0.0f);
  for (int d = 0; d < dim; ++d) {
    emb[static_cast<std::size_t>(3 * dim + d)] = 1.0f;
    outw[static_cast<std::size_t>(4 * dim + d)] = 1.0f;  // prefer token 4
  }

  std::unordered_map<std::string, std::pair<Shape, std::vector<float>>> tensors;
  tensors["embed.weight"] = {Shape{{vocab, dim}}, emb};
  tensors["norm.weight"] = {Shape{{dim}}, norm};
  tensors["lm_head.weight"] = {Shape{{vocab, dim}}, outw};
  return loaders::safetensors_write_f32(
      path, tensors, {{"format", "uaii-tiny"}, {"arch", "tiny_lm"}});
}

Error run_loaded_generate(ir::Graph graph,
                          float token_id,
                          std::string* decoded,
                          bool* ok,
                          const tokenizers::SimpleTokenizer& tok) {
  if (ok) *ok = false;
  SessionOptions opts;
  opts.validate = true;
  opts.weight_init = WeightInit::None;
  Session session;
  Error err = session.create(std::move(graph), opts);
  if (!err.ok()) return err;

  err = session.set_tensor_f32("tokens", {token_id});
  if (!err.ok()) return err;
  err = session.run();
  if (!err.ok()) return err;

  std::vector<float> probs;
  err = session.get_tensor_f32("probs", &probs);
  if (!err.ok()) return err;

  float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
  int argmax = 0;
  for (std::size_t i = 1; i < probs.size(); ++i) {
    if (probs[i] > probs[static_cast<std::size_t>(argmax)]) {
      argmax = static_cast<int>(i);
    }
  }

  std::string text;
  err = tok.decode({static_cast<std::int64_t>(argmax)}, &text);
  if (!err.ok()) return err;
  if (decoded) *decoded = text;

  const bool good = nearly_one(sum) && !probs.empty();
  if (ok) *ok = good;
  log::info("demo") << "generate probs_sum=" << sum << " argmax=" << argmax
                    << " decode='" << text << "'";
  if (!good) {
    return Error::make(ErrorCode::Internal, "generate demo failed probability check");
  }
  return Error::success();
}

}  // namespace

Error run_gguf_generate_demo(std::string* decoded, bool* ok) {
  const std::string dir = models_dir();
  const std::string path = dir + "/tiny_demo.gguf";
  Error err = write_tiny_lm_gguf(path, /*vocab=*/10, /*dim=*/4);
  if (!err.ok()) return err;

  ir::Graph graph;
  err = loaders::load_model(path, &graph);
  if (!err.ok()) return err;

  auto tok = tokenizers::SimpleTokenizer::demo_vocab();
  // Feed token id 3 ("hello")
  return run_loaded_generate(std::move(graph), 3.0f, decoded, ok, tok);
}

Error run_safetensors_generate_demo(std::string* decoded, bool* ok) {
  const std::string dir = models_dir();
  const std::string path = dir + "/tiny_demo.safetensors";
  Error err = write_tiny_lm_safetensors(path, /*vocab=*/10, /*dim=*/4);
  if (!err.ok()) return err;

  ir::Graph graph;
  err = loaders::load_model(path, &graph);
  if (!err.ok()) return err;

  auto tok = tokenizers::SimpleTokenizer::demo_vocab();
  return run_loaded_generate(std::move(graph), 3.0f, decoded, ok, tok);
}

Error run_moe_smoke_demo(bool* ok) {
  if (ok) *ok = false;

  // x[1,4] -> MoERouter(gate[2,4]) -> probs[1,2], top[1,1]
  //        -> MoEExperts(experts[2,4,4], top) -> y[1,4]
  ir::GraphBuilder b("moe_smoke");
  b.set_producer("uaii-phase4-demo");
  TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
  TensorId gate = b.add_weight("gate", DType::F32, Shape{{2, 4}}, "moe/gate.bin");
  TensorId probs = b.add_tensor("probs", DType::F32, Shape{{1, 2}});
  TensorId top = b.add_tensor("top_expert", DType::F32, Shape{{1, 1}});
  TensorId experts =
      b.add_weight("experts", DType::F32, Shape{{2, 4, 4}}, "moe/experts.bin");
  TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});

  b.add_node("router", "MoERouter", "1", {x, gate}, {probs, top},
             {ir::make_int_attr("num_experts", 2)});
  b.add_node("experts", "MoEExperts", "1", {x, experts, top}, {y},
             {ir::make_int_attr("num_experts", 2)});
  b.set_inputs({x}).set_outputs({y, probs, top});

  SessionOptions opts;
  opts.weight_init = WeightInit::Sequence;
  opts.validate = true;
  Session session;
  Error err = session.create(b.build(), opts);
  if (!err.ok()) return err;

  err = session.set_tensor_f32("x", {1.f, 0.f, 0.f, 0.f});
  if (!err.ok()) return err;
  err = session.run();
  if (!err.ok()) return err;

  std::vector<float> out_y;
  std::vector<float> out_p;
  std::vector<float> out_t;
  err = session.get_tensor_f32("y", &out_y);
  if (!err.ok()) return err;
  err = session.get_tensor_f32("probs", &out_p);
  if (!err.ok()) return err;
  err = session.get_tensor_f32("top_expert", &out_t);
  if (!err.ok()) return err;

  const float psum = out_p[0] + out_p[1];
  const bool good = out_y.size() == 4 && nearly_one(psum) && out_t.size() == 1 &&
                    (out_t[0] == 0.f || out_t[0] == 1.f);
  if (ok) *ok = good;
  log::info("demo") << "moe smoke probs=[" << out_p[0] << "," << out_p[1]
                    << "] top=" << out_t[0] << " y0=" << out_y[0];
  if (!good) {
    return Error::make(ErrorCode::Internal, "moe smoke failed");
  }
  return Error::success();
}

}  // namespace runtime
}  // namespace uaii
