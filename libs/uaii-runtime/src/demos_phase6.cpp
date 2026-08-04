#include "uaii/runtime/session.hpp"

#include "uaii/ir/graph.hpp"
#include "uaii/planner/optimize.hpp"
#include "uaii/quant/quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace uaii {
namespace runtime {
namespace {

constexpr const char* kPhase6Dir = "uaii_phase6_models";

void ensure_dir(const std::string& dir) {
#if defined(_WIN32)
  _mkdir(dir.c_str());
#else
  mkdir(dir.c_str(), 0755);
#endif
}

ir::Graph make_mlp_with_identity() {
  ir::GraphBuilder b("phase6_mlp");
  b.set_producer("uaii-phase6");
  const TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
  const TensorId w1 = b.add_weight("w1", DType::F32, Shape{{4, 8}}, "w1.bin");
  const TensorId h = b.add_tensor("h", DType::F32, Shape{{1, 8}});
  const TensorId h_id = b.add_tensor("h_id", DType::F32, Shape{{1, 8}});
  const TensorId h_act = b.add_tensor("h_act", DType::F32, Shape{{1, 8}});
  const TensorId w2 = b.add_weight("w2", DType::F32, Shape{{8, 4}}, "w2.bin");
  const TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
  const TensorId y_prob = b.add_tensor("y_prob", DType::F32, Shape{{1, 4}});

  b.add_node("fc1", "MatMul", "1", {x, w1}, {h});
  b.add_node("id", "Identity", "1", {h}, {h_id});
  b.add_node("act", "Relu", "1", {h_id}, {h_act});
  b.add_node("fc2", "MatMul", "1", {h_act, w2}, {y});
  b.add_node("prob", "Softmax", "1", {y}, {y_prob},
             {ir::make_int_attr("axis", -1)});
  b.set_inputs({x}).set_outputs({y_prob});
  return b.build();
}

Error write_f32_bin(const std::string& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return Error::make(ErrorCode::IoError, "write failed " + path);
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
  return Error::ok();
}

}  // namespace

Error run_optimize_demo(OptimizeDemoReport* report) {
  if (report == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "report null");
  }
  *report = OptimizeDemoReport{};

  const ir::Graph g = make_mlp_with_identity();

  planner::OptimizeOptions base;
  base.enable_fusion = false;
  base.enable_memory_reuse = true;
  base.enable_storage_plan = true;
  base.enable_plan_cache = false;
  planner::OptimizeResult r0;
  Error err = planner::optimize_graph(g, base, &r0);
  if (!err.ok()) return err;

  planner::OptimizeOptions opt;
  opt.enable_fusion = true;
  opt.enable_memory_reuse = true;
  opt.enable_storage_plan = true;
  opt.enable_plan_cache = true;
  planner::OptimizeResult r1;
  err = planner::optimize_graph(g, opt, &r1);
  if (!err.ok()) return err;

  report->baseline.summary = r0.summary;
  report->baseline.fusion = r0.fusion;
  report->baseline.memory = r0.memory;
  report->baseline.storage = r0.storage;

  report->optimized.summary = r1.summary;
  report->optimized.fusion = r1.fusion;
  report->optimized.memory = r1.memory;
  report->optimized.storage = r1.storage;
  report->optimized.cache_hit = r1.cache_hit;

  SessionOptions sopts;
  sopts.weight_init = WeightInit::Ones;
  sopts.enable_fusion = true;
  sopts.enable_memory_reuse = true;
  Session session;
  err = session.create(g, sopts);
  if (!err.ok()) return err;
  err = session.set_tensor_f32("x", {1.f, 2.f, 3.f, 4.f});
  if (!err.ok()) return err;
  err = session.run();
  if (!err.ok()) return err;
  std::vector<float> out;
  err = session.get_tensor_f32("y_prob", &out);
  if (!err.ok()) return err;

  const bool fused = r1.fusion.nodes_after < r0.fusion.nodes_after ||
                     r1.fusion.matmul_relu_fused > 0 || r1.fusion.identity_removed > 0;
  const bool mem_win =
      r1.memory.peak_bytes > 0 && r1.memory.peak_bytes <= r1.memory.naive_bytes;
  const bool correct = out.size() == 4 && std::fabs(out[0] - 0.25f) < 1e-3f;

  report->ok = fused && mem_win && correct;
  report->message = "baseline_nodes=" + std::to_string(r0.fusion.nodes_after) +
                    " optimized_nodes=" + std::to_string(r1.fusion.nodes_after) +
                    " naive_B=" + std::to_string(r1.memory.naive_bytes) +
                    " peak_B=" + std::to_string(r1.memory.peak_bytes) +
                    " fused=" + std::string(fused ? "yes" : "no") +
                    " correct=" + std::string(correct ? "yes" : "no");
  if (!report->ok) {
    return Error::make(ErrorCode::Internal, report->message);
  }
  return Error::ok();
}

Error run_streaming_demo(StreamingDemoReport* report) {
  if (report == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "report null");
  }
  *report = StreamingDemoReport{};

  ensure_dir(kPhase6Dir);
  const int dim = 256;
  const std::size_t n = static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim);
  std::vector<float> ones(n, 1.f);
  Error err = write_f32_bin(std::string(kPhase6Dir) + "/stream_w1.bin", ones);
  if (!err.ok()) return err;
  err = write_f32_bin(std::string(kPhase6Dir) + "/stream_w2.bin", ones);
  if (!err.ok()) return err;

  ir::GraphBuilder b("streaming_demo");
  b.set_producer("uaii-phase6");
  const TensorId x = b.add_tensor("x", DType::F32, Shape{{1, dim}});
  const TensorId wt1 =
      b.add_weight("w1", DType::F32, Shape{{dim, dim}}, "stream_w1.bin");
  const TensorId h = b.add_tensor("h", DType::F32, Shape{{1, dim}});
  const TensorId wt2 =
      b.add_weight("w2", DType::F32, Shape{{dim, dim}}, "stream_w2.bin");
  const TensorId y = b.add_tensor("y", DType::F32, Shape{{1, dim}});
  b.add_node("fc1", "MatMul", "1", {x, wt1}, {h});
  b.add_node("fc2", "MatMul", "1", {h, wt2}, {y});
  b.set_inputs({x}).set_outputs({y});

  const std::uint64_t weight_bytes =
      2ull * static_cast<std::uint64_t>(n) * sizeof(float);
  const std::uint64_t budget = weight_bytes / 2 + 1024;

  SessionOptions opts;
  opts.weights_dir = kPhase6Dir;
  opts.weight_init = WeightInit::None;
  opts.enable_fusion = false;
  opts.enable_memory_reuse = true;
  opts.enable_streaming = true;
  opts.enable_profiler = true;
  opts.allocator.budget_bytes = budget;

  Session session;
  err = session.create(b.build(), opts);
  if (!err.ok()) return err;

  std::vector<float> xin(static_cast<std::size_t>(dim), 0.f);
  xin[0] = 1.f;
  err = session.set_tensor_f32("x", xin);
  if (!err.ok()) return err;
  err = session.run();
  if (!err.ok()) return err;

  report->total_weight_bytes = session.optimize_report().storage.total_weight_bytes;
  report->ram_budget_bytes = budget;
  report->staging_bytes = session.optimize_report().storage.staging_bytes;
  report->bytes_read = report->total_weight_bytes;
  report->ok = session.optimize_report().storage.streaming_required &&
               report->staging_bytes > 0 &&
               report->staging_bytes < report->total_weight_bytes;
  report->message = "total_w=" + std::to_string(report->total_weight_bytes) +
                    " budget=" + std::to_string(report->ram_budget_bytes) +
                    " staging=" + std::to_string(report->staging_bytes) +
                    " streaming=" +
                    std::string(session.optimize_report().storage.streaming_required
                                    ? "yes"
                                    : "no");
  if (!report->ok) {
    return Error::make(ErrorCode::Internal, report->message);
  }
  return Error::ok();
}

Error run_profile_demo(const std::string& trace_path, ProfileDemoReport* report) {
  if (report == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "report null");
  }
  *report = ProfileDemoReport{};
  report->trace_path = trace_path.empty() ? "uaii_profile.json" : trace_path;

  SessionOptions opts;
  opts.weight_init = WeightInit::Ones;
  opts.enable_fusion = true;
  opts.enable_memory_reuse = true;
  opts.enable_profiler = true;
  opts.profile_trace_path = report->trace_path;

  Session session;
  Error err = session.create(make_mlp_with_identity(), opts);
  if (!err.ok()) return err;
  err = session.set_tensor_f32("x", {1.f, 2.f, 3.f, 4.f});
  if (!err.ok()) return err;
  err = session.run();
  if (!err.ok()) return err;

  report->summary = session.profiler().summary();
  report->ok = !session.profiler().events().empty();
  if (!report->ok) {
    return Error::make(ErrorCode::Internal, "no profiler events");
  }
  return Error::ok();
}

Error run_quant_demo(quant::QuantFormat format, QuantDemoReport* report) {
  if (report == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "report null");
  }
  *report = QuantDemoReport{};
  report->format = format;

  std::vector<float> src = {0.f, 0.5f, -0.25f, 1.f, -1.f, 0.125f, 0.75f, -0.5f};
  auto* q = quant::QuantizerRegistry::instance().default_quantizer();
  if (q == nullptr) {
    return Error::make(ErrorCode::Internal, "no quantizer registered");
  }
  quant::QuantParams params;
  params.group_size = 4;
  params.scale = 0.f;
  std::vector<std::uint8_t> packed;
  std::vector<float> scales;
  Error err = q->pack(src.data(), src.size(), format, params, &packed, &scales);
  if (!err.ok()) return err;
  std::vector<float> dst(src.size());
  err = q->unpack(packed.data(), packed.size(), src.size(), format, params,
                  scales.data(), scales.size(), dst.data());
  if (!err.ok()) return err;

  float max_err = 0.f;
  for (std::size_t i = 0; i < src.size(); ++i) {
    max_err = std::max(max_err, std::fabs(src[i] - dst[i]));
  }
  report->max_abs_err = max_err;
  float tol = 1e-3f;
  if (format == quant::QuantFormat::F16) tol = 1e-2f;
  if (format == quant::QuantFormat::BF16) tol = 5e-2f;
  if (format == quant::QuantFormat::INT8) tol = 0.05f;
  if (format == quant::QuantFormat::INT4 || format == quant::QuantFormat::NF4 ||
      format == quant::QuantFormat::MXFP4) {
    tol = 0.35f;
  }
  report->ok = max_err <= tol;
  report->message = std::string("format=") + quant::to_string(format) +
                    " max_abs_err=" + std::to_string(max_err) +
                    " tol=" + std::to_string(tol);
  if (!report->ok) {
    return Error::make(ErrorCode::Internal, report->message);
  }
  return Error::ok();
}

}  // namespace runtime
}  // namespace uaii
