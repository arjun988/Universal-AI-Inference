#include "commands/optimize.hpp"

#include "uaii/ir/graph.hpp"
#include "uaii/ir/serialize.hpp"
#include "uaii/planner/cache.hpp"
#include "uaii/planner/optimize.hpp"
#include "uaii/runtime/session.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>

namespace uaii {
namespace cli {
namespace {

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
  for (const auto& a : args) {
    if (a == flag) return true;
  }
  return false;
}

std::string get_opt(const std::vector<std::string>& args, const std::string& key,
                    const std::string& def = {}) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) return args[i + 1];
  }
  return def;
}

}  // namespace

int cmd_profile(const std::vector<std::string>& args) {
  if (has_flag(args, "--help") || has_flag(args, "-h")) {
    std::cout << "Usage:\n"
              << "  uaii profile --demo\n"
              << "  uaii profile <ir-path> --output trace.json [options]\n"
              << "Options:\n"
              << "  --output <path>     Chrome-trace JSON (default: uaii_profile.json)\n"
              << "  --weight-init ones  Weight init for missing files\n"
              << "  --no-fusion         Disable fusion\n";
    return 0;
  }

  if (get_opt(args, "--demo") == "1" || has_flag(args, "--demo") ||
      (!args.empty() && args[0] == "--demo")) {
    runtime::ProfileDemoReport report;
    const std::string out = get_opt(args, "--output", "uaii_profile.json");
    Error err = runtime::run_profile_demo(out, &report);
    std::cout << "trace: " << report.trace_path << '\n'
              << report.summary << '\n';
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
    return report.ok ? 0 : 2;
  }

  std::string ir_path;
  for (const auto& a : args) {
    if (!a.empty() && a[0] != '-') {
      ir_path = a;
      break;
    }
  }
  if (ir_path.empty()) {
    std::cerr << "uaii profile: missing ir path or --demo\n";
    return 1;
  }

  ir::Graph graph;
  Error err = ir::load_graph(ir_path, &graph);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  runtime::SessionOptions opts;
  opts.enable_profiler = true;
  opts.profile_trace_path = get_opt(args, "--output", "uaii_profile.json");
  opts.enable_fusion = !has_flag(args, "--no-fusion");
  opts.weight_init = runtime::WeightInit::Ones;
  if (get_opt(args, "--weight-init") == "zeros") opts.weight_init = runtime::WeightInit::Zeros;

  runtime::Session session;
  err = session.create(std::move(graph), opts);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }
  err = session.run();
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }
  std::cout << "trace: " << opts.profile_trace_path << '\n'
            << session.profiler().summary() << '\n';
  return 0;
}

int cmd_benchmark(const std::vector<std::string>& args) {
  if (has_flag(args, "--help") || has_flag(args, "-h") || args.empty()) {
    std::cout << "Usage:\n"
              << "  uaii benchmark --demo [ --iters N ]\n"
              << "Compares Phase 3-style baseline vs Phase 6 optimized path.\n";
    return args.empty() ? 1 : 0;
  }

  int iters = 20;
  try {
    iters = std::max(1, std::stoi(get_opt(args, "--iters", "20")));
  } catch (...) {
    iters = 20;
  }

  runtime::OptimizeDemoReport opt_report;
  Error err = runtime::run_optimize_demo(&opt_report);
  if (!err.ok() && !opt_report.ok) {
    // still print partial
  }

  auto time_ms = [&](bool fusion) {
    double total = 0;
    for (int i = 0; i < iters; ++i) {
      runtime::SessionOptions opts;
      opts.weight_init = runtime::WeightInit::Ones;
      opts.enable_fusion = fusion;
      opts.enable_memory_reuse = fusion;
      opts.enable_plan_cache = true;
      runtime::Session session;
      ir::GraphBuilder b("bench_mlp");
      // reuse optimize demo graph via session create from demos path:
      // build minimal each iter
      const TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
      const TensorId w1 = b.add_weight("w1", DType::F32, Shape{{4, 8}}, "w1.bin");
      const TensorId h = b.add_tensor("h", DType::F32, Shape{{1, 8}});
      const TensorId h_act = b.add_tensor("h_act", DType::F32, Shape{{1, 8}});
      const TensorId w2 = b.add_weight("w2", DType::F32, Shape{{8, 4}}, "w2.bin");
      const TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
      const TensorId yp = b.add_tensor("y_prob", DType::F32, Shape{{1, 4}});
      b.add_node("fc1", "MatMul", "1", {x, w1}, {h});
      b.add_node("act", "Relu", "1", {h}, {h_act});
      b.add_node("fc2", "MatMul", "1", {h_act, w2}, {y});
      b.add_node("prob", "Softmax", "1", {y}, {yp}, {ir::make_int_attr("axis", -1)});
      b.set_inputs({x}).set_outputs({yp});

      Error e = session.create(b.build(), opts);
      if (!e.ok()) return -1.0;
      e = session.set_tensor_f32("x", {1.f, 2.f, 3.f, 4.f});
      if (!e.ok()) return -1.0;
      const auto t0 = std::chrono::steady_clock::now();
      e = session.run();
      const auto t1 = std::chrono::steady_clock::now();
      if (!e.ok()) return -1.0;
      total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / static_cast<double>(iters);
  };

  const double base_ms = time_ms(false);
  const double opt_ms = time_ms(true);

  std::cout << "optimize: " << opt_report.message << '\n';
  std::cout << "benchmark iters=" << iters << '\n';
  std::cout << "  baseline_ms=" << base_ms << '\n';
  std::cout << "  optimized_ms=" << opt_ms << '\n';
  if (base_ms > 0 && opt_ms > 0) {
    std::cout << "  speedup_x=" << (base_ms / opt_ms) << '\n';
  }
  std::cout << "  baseline_mem: " << opt_report.baseline.memory.summary << '\n';
  std::cout << "  optimized_mem: " << opt_report.optimized.memory.summary << '\n';
  return (base_ms > 0 && opt_ms > 0 && opt_report.ok) ? 0 : 2;
}

int cmd_cache(const std::vector<std::string>& args) {
  if (has_flag(args, "--help") || has_flag(args, "-h")) {
    std::cout << "Usage:\n"
              << "  uaii cache status\n"
              << "  uaii cache clear\n";
    return 0;
  }
  const std::string sub = args.empty() ? "status" : args[0];
  if (sub == "clear") {
    planner::PlanCache::instance().clear();
    std::cout << "plan cache cleared\n";
    return 0;
  }
  if (sub == "status") {
    std::cout << "plan_cache_entries=" << planner::PlanCache::instance().size() << '\n';
    return 0;
  }
  std::cerr << "unknown cache subcommand\n";
  return 1;
}

}  // namespace cli
}  // namespace uaii
