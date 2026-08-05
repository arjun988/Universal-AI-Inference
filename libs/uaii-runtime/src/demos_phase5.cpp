#include "uaii/runtime/session.hpp"

#include "uaii/backends/cuda_backend.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"

#include <sstream>
#include <unordered_map>

namespace uaii {
namespace runtime {
namespace {

bool parity_force_host_fallback(const std::string& backend_name) {
  if (backend_name == "cuda") {
    return !(backends::CudaBackend::native_compiled() &&
             backends::CudaBackend::native_device_available());
  }
  return true;
}

bool parity_skip_gpu_backend(const std::string& backend_name, backends::ParityReport* report) {
  if (backend_name != "cuda") {
    return false;
  }
  if (!backends::CudaBackend::native_compiled()) {
    return false;
  }
  if (backends::CudaBackend::native_device_available()) {
    return false;
  }
  if (report != nullptr) {
    report->ok = true;
    report->message = "skipped: no CUDA device";
  }
  return true;
}

ir::Graph make_toy_mlp_parity_graph() {
  ir::GraphBuilder b("parity_toy_mlp");
  b.set_producer("uaii-phase5-parity");

  const TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}}, ir::StorageHint::Ram);
  const TensorId w1 =
      b.add_weight("w1", DType::F32, Shape{{4, 8}}, "weights/w1.bin");
  const TensorId h = b.add_tensor("h", DType::F32, Shape{{1, 8}});
  const TensorId h_act = b.add_tensor("h_act", DType::F32, Shape{{1, 8}});
  const TensorId w2 =
      b.add_weight("w2", DType::F32, Shape{{8, 4}}, "weights/w2.bin");
  const TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
  const TensorId y_prob = b.add_tensor("y_prob", DType::F32, Shape{{1, 4}});

  b.add_node("fc1", "MatMul", "1", {x, w1}, {h});
  b.add_node("act", "Relu", "1", {h}, {h_act});
  b.add_node("fc2", "MatMul", "1", {h_act, w2}, {y});
  b.add_node("prob", "Softmax", "1", {y}, {y_prob},
             {ir::make_int_attr("axis", -1)});
  b.set_inputs({x}).set_outputs({y_prob});
  return b.build();
}

Error run_once(const ir::Graph& graph,
               const std::string& backend_name,
               const std::vector<float>& input_f32,
               const std::string& input_name,
               std::unordered_map<std::string, std::vector<float>>* outs,
               bool force_host_fallback) {
  SessionOptions opts;
  opts.weight_init = WeightInit::Ones;
  opts.validate = true;
  opts.backend_name = backend_name;
  opts.force_host_fallback = force_host_fallback;
  opts.prefer_native = !force_host_fallback;

  Session session;
  Error err = session.create(graph, opts);
  if (!err.ok()) {
    return err;
  }
  err = session.set_tensor_f32(input_name, input_f32);
  if (!err.ok()) {
    return err;
  }
  err = session.run();
  if (!err.ok()) {
    return err;
  }

  outs->clear();
  for (TensorId oid : session.graph().outputs) {
    const auto* t = session.graph().find_tensor(oid);
    const std::string name =
        (t && !t->name.empty()) ? t->name : ("#" + std::to_string(oid));
    std::vector<float> values;
    err = session.get_tensor_f32(name, &values);
    if (!err.ok()) {
      return err;
    }
    (*outs)[name] = std::move(values);
  }
  return Error::success();
}

}  // namespace

Error run_backend_parity(const ir::Graph& graph,
                         const std::string& backend_a,
                         const std::string& backend_b,
                         const backends::ParityPolicy& policy,
                         backends::ParityReport* report,
                         const std::vector<float>& input_f32,
                         const std::string& input_name) {
  if (report == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "parity report null");
  }
  *report = backends::ParityReport{};
  report->backend_a = backend_a;
  report->backend_b = backend_b;

  std::unordered_map<std::string, std::vector<float>> outs_a;
  std::unordered_map<std::string, std::vector<float>> outs_b;

  if (parity_skip_gpu_backend(backend_a, report) || parity_skip_gpu_backend(backend_b, report)) {
    log::info("parity") << report->backend_a << " vs " << report->backend_b << ": "
                        << report->message;
    return Error::success();
  }

  const bool fb_a = parity_force_host_fallback(backend_a);
  const bool fb_b = parity_force_host_fallback(backend_b);

  Error err = run_once(graph, backend_a, input_f32, input_name, &outs_a, fb_a);
  if (!err.ok()) {
    report->ok = false;
    report->message = "backend_a failed: " + err.to_string();
    return err;
  }
  err = run_once(graph, backend_b, input_f32, input_name, &outs_b, fb_b);
  if (!err.ok()) {
    report->ok = false;
    report->message = "backend_b failed: " + err.to_string();
    return err;
  }

  if (outs_a.size() != outs_b.size()) {
    report->ok = false;
    report->message = "output count mismatch";
    return Error::make(ErrorCode::InvalidArgument, report->message);
  }

  bool all_ok = true;
  for (const auto& kv : outs_a) {
    auto it = outs_b.find(kv.first);
    if (it == outs_b.end()) {
      report->ok = false;
      report->message = "missing output on backend_b: " + kv.first;
      return Error::make(ErrorCode::NotFound, report->message);
    }
    if (kv.second.size() != it->second.size()) {
      report->ok = false;
      report->message = "size mismatch for " + kv.first;
      return Error::make(ErrorCode::InvalidArgument, report->message);
    }
    backends::ParityTensorDiff diff;
    diff.name = kv.first;
    err = backends::compare_f32_buffers(kv.second.data(), it->second.data(),
                                        kv.second.size(), policy, &diff);
    if (!err.ok()) {
      return err;
    }
    report->diffs.push_back(diff);
    if (!diff.ok) {
      all_ok = false;
    }
  }

  report->ok = all_ok;
  if (all_ok) {
    report->message = "parity ok under atol=" + std::to_string(policy.atol) +
                      " rtol=" + std::to_string(policy.rtol);
  } else {
    std::ostringstream oss;
    oss << "parity failed:";
    for (const auto& d : report->diffs) {
      if (!d.ok) {
        oss << " " << d.name << "(max_abs=" << d.max_abs_diff
            << ", max_rel=" << d.max_rel_diff << ")";
      }
    }
    report->message = oss.str();
  }

  log::info("parity") << report->backend_a << " vs " << report->backend_b << ": "
                      << report->message;
  return Error::success();
}

Error run_parity_demo(backends::ParityReport* report) {
  backends::ParityReport local;
  backends::ParityReport* out = report ? report : &local;
  backends::ParityPolicy policy;
  policy.atol = 1e-5f;
  policy.rtol = 1e-4f;
  policy.require_finite = true;
  policy.outputs_only = true;

  const ir::Graph graph = make_toy_mlp_parity_graph();
  Error err = run_backend_parity(graph, "cpu", "cuda", policy, out,
                                 {1.f, 2.f, 3.f, 4.f}, "x");
  if (!err.ok()) {
    return err;
  }
  if (!out->ok) {
    return Error::make(ErrorCode::Internal, out->message);
  }
  return Error::success();
}

}  // namespace runtime
}  // namespace uaii
