#include "uaii/ir/graph.hpp"
#include "uaii/planner/storage_plan.hpp"
#include "uaii/quant/formats.hpp"

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

  {
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
    opts.ram_budget_bytes = 32;  // force stream (weights are larger); < 1 MiB keeps always-pin
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
  }

  // Tight laptop-class budget: unpin embed/lm_head rather than OOM (kimi floor dial).
  {
    ir::GraphBuilder b("tight");
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
    // 512*512 f32 = 1 MiB each. Total ~4.06 MiB > 4 MiB budget.
    TensorId emb =
        b.add_weight("model.embed_tokens.weight", DType::F32, Shape{{512, 512}}, "e.bin");
    TensorId lm = b.add_weight("lm_head.weight", DType::F32, Shape{{512, 512}}, "lm.bin");
    TensorId l0 = b.add_weight("model.layers.0.self_attn.q_proj.weight", DType::F32,
                               Shape{{64, 64}}, "l0.bin");
    TensorId l1 = b.add_weight("model.layers.1.self_attn.q_proj.weight", DType::F32,
                               Shape{{512, 512}}, "l1.bin");
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
    b.add_node("id", "Identity", "1", {x}, {y});
    (void)emb;
    (void)lm;
    (void)l0;
    (void)l1;
    ir::Graph g = b.build();

    planner::StoragePlanOptions opts;
    opts.enable_streaming = true;
    opts.ram_budget_bytes = (5ull * 1024ull * 1024ull) / 2;  // 2.5 MiB: unpin embed and lm_head
    opts.pin_layers = 2;
    opts.expert_cache_bytes = 2ull * 1024ull * 1024ull;  // leftover after trunk, not stolen from it
    planner::StoragePlan plan;
    const Error err = planner::build_storage_plan(g, opts, &plan);
    expect(err.ok(), "tight budget plan");
    expect(plan.streaming_required, "tight streaming_required");

    bool pin_emb = true, pin_lm = true, stream_l1 = false;
    for (const auto& p : plan.placements) {
      if (p.name.find("embed") != std::string::npos) pin_emb = !p.stream;
      if (p.name.find("lm_head") != std::string::npos) pin_lm = !p.stream;
      if (p.name.find("layers.1.") != std::string::npos) stream_l1 = p.stream;
    }
    expect(!pin_emb, "tight: embeddings streamed");
    expect(!pin_lm, "tight: lm_head streamed");
    expect(stream_l1, "tight: later layer streamed");
    expect(plan.summary.find("pin_embed=no") != std::string::npos, "summary pin_embed=no");
    expect(plan.summary.find("pin_lm_head=no") != std::string::npos, "summary pin_lm_head=no");
    expect(plan.expert_cache_bytes < (2ull * 1024ull * 1024ull),
           "expert LRU is leftover after trunk");
  }

  // Packed GGUF size is the resident/staging size, not widened f32.
  {
    ir::GraphBuilder b("q4");
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 256}});
    const Shape wshape{{256, 256}};
    const std::size_t n = 256ull * 256ull;
    TensorId w = b.add_weight("blk.0.attn_q.weight", DType::F32, wshape, "m.gguf#blk.0.attn_q.weight",
                              quant::QuantFormat::Q4_K);
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 256}});
    b.add_node("id", "Identity", "1", {x}, {y});
    (void)w;
    ir::Graph g = b.build();

    planner::StoragePlanOptions opts;
    opts.enable_streaming = true;
    opts.ram_budget_bytes = 8ull * 1024ull * 1024ull;
    opts.pin_layers = 0;
    planner::StoragePlan plan;
    const Error err = planner::build_storage_plan(g, opts, &plan);
    expect(err.ok(), "q4 plan");
    const std::uint64_t packed =
        static_cast<std::uint64_t>(quant::packed_nbytes(quant::QuantFormat::Q4_K, n));
    const std::uint64_t f32n = static_cast<std::uint64_t>(n) * 4ull;
    expect(packed > 0 && packed < f32n, "q4 packed smaller than f32");
    bool found = false;
    for (const auto& p : plan.placements) {
      if (p.name.find("attn_q") != std::string::npos) {
        found = true;
        expect(p.bytes == packed, "q4 placement uses packed bytes");
      }
    }
    expect(found, "q4 weight placed");
    expect(plan.total_weight_bytes == packed, "q4 total_weight_bytes packed");
  }

  // A single f32 tensor larger than the budget fail-closes instead of silent OOM.
  {
    ir::GraphBuilder b("too_big");
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
    TensorId w =
        b.add_weight("model.layers.3.self_attn.q_proj.weight", DType::F32, Shape{{1024, 1024}},
                     "big.bin");  // 4 MiB
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
    b.add_node("id", "Identity", "1", {x}, {y});
    (void)w;
    ir::Graph g = b.build();

    planner::StoragePlanOptions opts;
    opts.enable_streaming = true;
    opts.ram_budget_bytes = 1ull * 1024ull * 1024ull;  // 1 MiB; 2×staging = 8 MiB
    opts.pin_layers = 0;
    planner::StoragePlan plan;
    const Error err = planner::build_storage_plan(g, opts, &plan);
    expect(!err.ok(), "oversized f32 tensor fail-closed");
    expect(std::string(err.message()).find("exceeds the memory preset") != std::string::npos,
           "oversized message");
  }

  // One staging slot fits when pin + 1×stage fits and pin + 2×stage does not.
  // Two 1 MiB layers (2 MiB total) exceed a 1.5 MiB budget, so streaming turns on.
  {
    ir::GraphBuilder b("one_slot");
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
    TensorId w3 = b.add_weight("model.layers.3.self_attn.q_proj.weight", DType::F32,
                               Shape{{512, 512}}, "l3.bin");  // 1 MiB
    TensorId w4 = b.add_weight("model.layers.4.self_attn.q_proj.weight", DType::F32,
                               Shape{{512, 512}}, "l4.bin");  // 1 MiB
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
    b.add_node("id", "Identity", "1", {x}, {y});
    (void)w3;
    (void)w4;
    ir::Graph g = b.build();

    planner::StoragePlanOptions wide;
    wide.enable_streaming = true;
    wide.ram_budget_bytes = (3ull * 1024ull * 1024ull) / 2ull;  // 1.5 MiB
    wide.pin_layers = 0;
    wide.staging_slots = 2;
    planner::StoragePlan wide_plan;
    const Error wide_err = planner::build_storage_plan(g, wide, &wide_plan);
    expect(!wide_err.ok(), "double staging exceeds 1.5 MiB");

    planner::StoragePlanOptions one = wide;
    one.staging_slots = 1;
    planner::StoragePlan plan;
    const Error err = planner::build_storage_plan(g, one, &plan);
    expect(err.ok(), "single staging fits 1.5 MiB");
    expect(plan.staging_slots == 1, "staging_slots=1");
    expect(plan.streaming_required, "layer streamed");
    int streamed = 0;
    for (const auto& p : plan.placements) {
      if (p.stream) ++streamed;
    }
    expect(streamed == 2, "both 1 MiB layers streamed into one slot");
  }

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_storage_plan_test: OK\n";
  return EXIT_SUCCESS;
}
