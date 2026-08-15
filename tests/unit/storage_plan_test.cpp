#include "uaii/ir/graph.hpp"
#include "uaii/planner/storage_plan.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  using namespace uaii;

  expect(planner::is_moe_expert_weight_name("blk.3.ffn_gate_exps.weight"), "gguf exps");
  expect(planner::is_moe_expert_weight_name("blk.0.exp.3.ffn_gate.weight"), "gguf listed expert");
  expect(planner::is_moe_expert_weight_name("model.layers.0.block_sparse_moe.experts.2.w1.weight"),
         "hf mixtral expert");
  expect(!planner::is_moe_expert_weight_name("model.layers.0.self_attn.q_proj.weight"),
         "dense attn not expert");
  expect(!planner::is_moe_expert_weight_name("block_sparse_moe.gate.weight"), "router not expert");

  ir::GraphBuilder b("mem");
  TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
  TensorId emb = b.add_weight("model.embed_tokens.weight", DType::F32, Shape{{8, 4}}, "e.bin");
  TensorId l0 = b.add_weight("model.layers.0.self_attn.q_proj.weight", DType::F32, Shape{{4, 4}},
                             "l0.bin");
  TensorId l1 = b.add_weight("model.layers.1.self_attn.q_proj.weight", DType::F32, Shape{{4, 4}},
                             "l1.bin");
  TensorId ex = b.add_weight("model.layers.1.block_sparse_moe.experts.0.w1.weight", DType::F32,
                             Shape{{8, 4}}, "e0.bin");
  TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
  b.add_node("id", "Identity", "1", {x}, {y});
  (void)emb;
  (void)l0;
  (void)l1;
  (void)ex;
  ir::Graph g = b.build();

  planner::StoragePlanOptions opts;
  opts.enable_streaming = true;
  opts.ram_budget_bytes = 32;  // force stream (weights are larger)
  opts.pin_layers = 1;
  opts.expert_cache_bytes = 1024;
  planner::StoragePlan plan;
  const Error err = planner::build_storage_plan(g, opts, &plan);
  expect(err.ok(), "build_storage_plan");
  expect(plan.streaming_required, "streaming_required");

  bool pinned_l0 = false, streamed_l1 = false, expert_stream = false, pin_emb = false;
  for (const auto& p : plan.placements) {
    if (p.name.find("embed") != std::string::npos) pin_emb = !p.stream;
    if (p.name.find("layers.0.") != std::string::npos && !p.expert) pinned_l0 = !p.stream;
    if (p.name.find("layers.1.self_attn") != std::string::npos) streamed_l1 = p.stream;
    if (p.expert) expert_stream = p.stream;
  }
  expect(pin_emb, "embeddings pinned");
  expect(pinned_l0, "layer 0 pinned");
  expect(streamed_l1, "layer 1 streamed");
  expect(expert_stream, "experts streamed not pinned");

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_storage_plan_test: OK\n";
  return EXIT_SUCCESS;
}
