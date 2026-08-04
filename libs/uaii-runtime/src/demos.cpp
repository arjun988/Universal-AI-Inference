#include "uaii/runtime/session.hpp"

#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"

#include <cmath>

namespace uaii {
namespace runtime {
namespace {

bool nearly_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

}  // namespace

Error run_toy_mlp_demo(std::vector<float>* out_values, bool* matched_expected) {
  if (matched_expected) {
    *matched_expected = false;
  }

  // x[1,4] @ w1[4,8] -> h; Relu; h @ w2[8,4] -> y; Softmax -> y_prob
  ir::GraphBuilder b("toy_mlp_demo");
  b.set_producer("uaii-phase3-demo");

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

  SessionOptions opts;
  opts.weight_init = WeightInit::Ones;  // all-ones weights for deterministic demo
  opts.validate = true;

  Session session;
  Error err = session.create(b.build(), opts);
  if (!err.ok()) {
    return err;
  }

  // Input: [1, 2, 3, 4]
  err = session.set_tensor_f32("x", {1.f, 2.f, 3.f, 4.f});
  if (!err.ok()) {
    return err;
  }

  err = session.run();
  if (!err.ok()) {
    return err;
  }

  std::vector<float> out;
  err = session.get_tensor_f32("y_prob", &out);
  if (!err.ok()) {
    return err;
  }
  if (out_values) {
    *out_values = out;
  }

  // With all-ones weights:
  // h = [10,10,10,10,10,10,10,10], relu same
  // y = [80,80,80,80], softmax -> uniform 0.25
  const bool ok = out.size() == 4 && nearly_equal(out[0], 0.25f) &&
                  nearly_equal(out[1], 0.25f) && nearly_equal(out[2], 0.25f) &&
                  nearly_equal(out[3], 0.25f);
  if (matched_expected) {
    *matched_expected = ok;
  }
  if (!ok) {
    return Error::make(ErrorCode::Internal,
                       "toy_mlp_demo output mismatch (expected uniform 0.25)");
  }
  log::info("demo") << "toy_mlp OK y_prob=[" << out[0] << ", " << out[1] << ", "
                    << out[2] << ", " << out[3] << "]";
  return Error::ok();
}

Error run_tiny_block_demo(std::vector<float>* out_values) {
  // Minimal transformer-style block:
  // x -> LayerNorm -> MatMul(attn) -> Softmax -> MatMul(out) -> Add(residual) -> RMSNorm
  ir::GraphBuilder b("tiny_block_demo");
  b.set_producer("uaii-phase3-demo");

  const TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
  const TensorId ln_w =
      b.add_weight("ln_w", DType::F32, Shape{{4}}, "weights/ln_w.bin");
  const TensorId ln_b =
      b.add_weight("ln_b", DType::F32, Shape{{4}}, "weights/ln_b.bin");
  const TensorId x_n = b.add_tensor("x_n", DType::F32, Shape{{1, 4}});

  const TensorId w_q =
      b.add_weight("w_q", DType::F32, Shape{{4, 4}}, "weights/w_q.bin");
  const TensorId q = b.add_tensor("q", DType::F32, Shape{{1, 4}});
  const TensorId attn = b.add_tensor("attn", DType::F32, Shape{{1, 4}});

  const TensorId w_o =
      b.add_weight("w_o", DType::F32, Shape{{4, 4}}, "weights/w_o.bin");
  const TensorId proj = b.add_tensor("proj", DType::F32, Shape{{1, 4}});
  const TensorId resid = b.add_tensor("resid", DType::F32, Shape{{1, 4}});

  const TensorId rms_w =
      b.add_weight("rms_w", DType::F32, Shape{{4}}, "weights/rms_w.bin");
  const TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});

  b.add_node("ln", "LayerNorm", "1", {x, ln_w, ln_b}, {x_n},
             {ir::make_float_attr("eps", 1e-5)});
  b.add_node("q_proj", "MatMul", "1", {x_n, w_q}, {q});
  b.add_node("sm", "Softmax", "1", {q}, {attn}, {ir::make_int_attr("axis", -1)});
  b.add_node("o_proj", "MatMul", "1", {attn, w_o}, {proj});
  b.add_node("residual", "Add", "1", {x, proj}, {resid});
  b.add_node("rms", "RMSNorm", "1", {resid, rms_w}, {y},
             {ir::make_float_attr("eps", 1e-5)});
  b.set_inputs({x}).set_outputs({y});

  SessionOptions opts;
  opts.weight_init = WeightInit::Ones;
  Session session;
  Error err = session.create(b.build(), opts);
  if (!err.ok()) {
    return err;
  }

  err = session.set_tensor_f32("x", {1.f, 0.f, -1.f, 0.5f});
  if (!err.ok()) {
    return err;
  }
  err = session.run();
  if (!err.ok()) {
    return err;
  }

  std::vector<float> out;
  err = session.get_tensor_f32("y", &out);
  if (!err.ok()) {
    return err;
  }
  if (out_values) {
    *out_values = out;
  }
  if (out.size() != 4) {
    return Error::make(ErrorCode::Internal, "tiny_block unexpected output size");
  }
  log::info("demo") << "tiny_block OK y=[" << out[0] << ", " << out[1] << ", "
                    << out[2] << ", " << out[3] << "]";
  return Error::ok();
}

}  // namespace runtime
}  // namespace uaii
