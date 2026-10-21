#include "uaii/ir/graph.hpp"
#include "uaii/kernels/thread_pool.hpp"
#include "uaii/runtime/memory_preset.hpp"

#include <string>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

}  // namespace

int main() {
  using namespace uaii::runtime;

  expect(clamp_budget_to_available(8ull * kGiB, 6ull * kGiB) == 3ull * kGiB,
         "8 GiB request on 6 GiB free clamps to 3 GiB");
  expect(clamp_budget_to_available(2ull * kGiB, 16ull * kGiB) == 2ull * kGiB,
         "2 GiB request under a 16 GiB free pool stays 2 GiB");
  expect(clamp_budget_to_available(2ull * kGiB, 0) == 2ull * kGiB,
         "unknown available RAM leaves the request");

  {
    MemoryPresetConfig cfg;
    cfg.preset = MemoryPreset::Phone;
    cfg.ram_budget_bytes = 8ull * kGiB;
    cfg.kv_window = 512;
    cfg.expert_cache_bytes = 256ull * kMiB;
    apply_available_memory_clamp(&cfg, 6ull * kGiB);
    expect(cfg.budget_clamped, "8 vs 6 is clamped");
    expect(cfg.ram_budget_bytes == 3ull * kGiB, "cap is 3 GiB");
    expect(cfg.kv_window == 512, "3 GiB keeps the 512 window");
    expect(cfg.expert_cache_bytes == 256ull * kMiB, "expert cache still under cap/8");
  }
  {
    MemoryPresetConfig cfg;
    cfg.preset = MemoryPreset::Phone;
    cfg.ram_budget_bytes = 2ull * kGiB;
    cfg.kv_window = 512;
    cfg.expert_cache_bytes = 256ull * kMiB;
    apply_available_memory_clamp(&cfg, 2ull * kGiB);
    expect(cfg.ram_budget_bytes == kGiB, "half of 2 GiB is 1 GiB");
    expect(cfg.kv_window == 256, "1 GiB uses the 256 window");
    expect(cfg.expert_cache_bytes == kGiB / 8ull, "expert cache capped at cap/8");
  }
  {
    MemoryPresetConfig cfg;
    cfg.preset = MemoryPreset::Phone;
    cfg.ram_budget_bytes = 2ull * kGiB;
    cfg.kv_window = 512;
    cfg.expert_cache_bytes = 256ull * kMiB;
    apply_available_memory_clamp(&cfg, (3ull * kGiB) / 2ull);
    expect(cfg.ram_budget_bytes == (3ull * kGiB) / 4ull, "half of 1.5 GiB is 0.75 GiB");
    expect(cfg.kv_window == 128, "under 1 GiB uses the 128 window");
  }
  {
    MemoryPresetConfig cfg;
    cfg.preset = MemoryPreset::Phone;
    cfg.ram_budget_bytes = 2ull * kGiB;
    cfg.kv_window = 512;
    apply_available_memory_clamp(&cfg, 16ull * kGiB);
    expect(!cfg.budget_clamped && cfg.ram_budget_bytes == 2ull * kGiB, "16 GiB free does not clamp");
    apply_available_memory_clamp(&cfg, 0);
    expect(!cfg.budget_clamped && cfg.ram_budget_bytes == 2ull * kGiB, "available 0 does not clamp");
  }
  {
    MemoryPresetConfig cfg;
    cfg.preset = MemoryPreset::Laptop;
    cfg.ram_budget_bytes = 8ull * kGiB;
    cfg.kv_window = 2048;
    cfg.expert_cache_bytes = kGiB;
    apply_available_memory_clamp(&cfg, 12ull * kGiB);
    expect(!cfg.budget_clamped, "laptop 8 GiB fits in 12 GiB free");
    expect(cfg.ram_budget_bytes == 8ull * kGiB, "laptop target stays 8 GiB");
    expect(cfg.kv_window == 2048, "laptop kv stays 2048");
    expect(cfg.expert_cache_bytes == kGiB, "laptop expert cache stays 1 GiB");
    apply_available_memory_clamp(&cfg, 6ull * kGiB);
    expect(cfg.budget_clamped, "laptop shrinks when free RAM is below the target");
    expect(cfg.ram_budget_bytes == 6ull * kGiB, "laptop cap is free RAM, not half");
    expect(cfg.kv_window == 2048, "6 GiB keeps the laptop window");
    expect(cfg.expert_cache_bytes == (6ull * kGiB) / 8ull, "expert cache follows the shrunk cap");
  }

  MemoryPreset parsed = MemoryPreset::None;
  expect(parse_memory_preset("phone", &parsed) && parsed == MemoryPreset::Phone, "parse phone");
  expect(!parse_memory_preset("tablet", &parsed), "unknown preset rejected");

  MemoryPresetConfig phone;
  const uaii::Error err = resolve_memory_preset(MemoryPreset::Phone, 0, &phone);
  expect(err.ok(), "resolve phone");
  expect(phone.label == "phone", "label");
  expect(phone.pin_layers == 0, "pin_layers 0");
  expect(phone.kv_sink == 4, "kv sink");
  expect(phone.single_staging, "single staging");
  expect(phone.max_threads == 4, "max threads 4");
  expect(phone.enable_streaming, "streaming on");

  const std::uint64_t avail = system_available_bytes();
  const std::uint64_t budget = clamp_budget_to_available(2ull * kGiB, avail);
  expect(phone.ram_budget_bytes == budget, "phone budget is min(2 GiB, available/2)");
  expect(phone.budget_clamped == (avail != 0 && budget < 2ull * kGiB), "clamped flag");
  std::int64_t expect_kv = 512;
  if (phone.budget_clamped && budget < kGiB) {
    expect_kv = 128;
  } else if (phone.budget_clamped && budget < kGiB + kGiB / 2ull) {
    expect_kv = 256;
  }
  expect(phone.kv_window == expect_kv, "phone kv window follows the clamp");
  const std::uint64_t expect_expert =
      phone.budget_clamped && (256ull * kMiB) > budget / 8ull ? budget / 8ull : 256ull * kMiB;
  expect(phone.expert_cache_bytes == expect_expert, "phone expert cache");

  SessionOptions opts;
  apply_memory_preset(phone, &opts);
  expect(opts.single_staging, "session single staging");
  expect(opts.max_threads == 4, "session threads");
  expect(opts.allocator.budget_bytes == phone.ram_budget_bytes, "session budget");
  expect(uaii::kernels::hardware_concurrency() <= 4u, "thread cap is 4");

  MemoryPresetConfig laptop;
  expect(resolve_memory_preset(MemoryPreset::Laptop, 0, &laptop).ok(), "resolve laptop");
  expect(!laptop.single_staging, "laptop preset keeps two staging slots");
  expect(laptop.max_threads == 0, "laptop preset has no thread cap");
  if (avail == 0 || avail >= 8ull * kGiB) {
    expect(laptop.ram_budget_bytes == 8ull * kGiB, "laptop target stays 8 GiB when free RAM covers it");
    expect(!laptop.budget_clamped, "laptop is not half-clamped");
    expect(laptop.kv_window == 2048, "laptop kv window");
    expect(laptop.expert_cache_bytes == kGiB, "laptop expert cache");
  } else {
    expect(laptop.budget_clamped, "laptop shrinks only below the target");
    expect(laptop.ram_budget_bytes == avail, "laptop cap is free RAM");
  }
  {
    using namespace uaii;
    ir::GraphBuilder b("pause");
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
    TensorId w1 = b.add_weight("w1", DType::F32, Shape{{4, 4}}, "missing.bin");
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
    b.add_node("m", "MatMul", "1", {x, w1}, {y});
    b.set_inputs({x}).set_outputs({y});
    SessionOptions ready;
    ready.weight_init = WeightInit::Ones;
    Session session;
    expect(session.create(b.build(), ready).ok(), "session for pause and cancel");
    session.set_background(true);
    const uaii::Error paused = session.run();
    expect(!paused.ok(), "background pauses run");
    expect(paused.message().find("paused: app backgrounded") != std::string::npos, "pause message");
    session.set_background(false);
    expect(session.set_tensor_f32("x", {1.f, 0.f, 0.f, 0.f}).ok(), "set x");
    expect(session.run().ok(), "run after foreground");
    std::atomic<bool> stop{false};
    std::thread warner([&] {
      while (!stop.load(std::memory_order_acquire)) session.on_memory_warning();
    });
    bool runs_ok = true;
    for (int i = 0; i < 40; ++i) {
      if (!session.run().ok()) runs_ok = false;
    }
    stop.store(true, std::memory_order_release);
    warner.join();
    expect(runs_ok, "run stays up while memory warnings arrive");
    session.request_cancel();
    const uaii::Error cancelled = session.run();
    expect(!cancelled.ok(), "cancel stops run");
    expect(cancelled.message().find("cancelled") != std::string::npos, "cancel message");
    session.on_memory_warning();
  }

  apply_memory_preset(laptop, &opts);
  expect(!opts.single_staging, "laptop keeps prefetch staging");
  expect(opts.max_threads == 0, "laptop clears the thread cap field");
  uaii::kernels::set_thread_limit(0);

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_memory_preset_test: OK\n";
  return EXIT_SUCCESS;
}
