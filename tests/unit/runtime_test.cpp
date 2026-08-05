#include "uaii/backends/parity.hpp"
#include "uaii/c_api/uaii.h"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/validator.hpp"
#include "uaii/kernels/kernels.hpp"
#include "uaii/kernels/view_util.hpp"
#include "uaii/plugins/operator_host.hpp"
#include "uaii/runtime/session.hpp"

#include <cmath>
#include <iostream>
#include <vector>

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
  // MatMul nbytes guard
  {
    float a[4] = {1, 2, 3, 4};
    float b[4] = {1, 0, 0, 1};
    float c[4] = {};
    std::int64_t sh[2] = {2, 2};
    uaii::kernels::TensorView A{uaii::DType::F32, sh, 2, a, sizeof(a)};
    uaii::kernels::TensorView B{uaii::DType::F32, sh, 2, b, sizeof(b)};
    uaii::kernels::TensorView C{uaii::DType::F32, sh, 2, c, sizeof(c)};
    expect(uaii::kernels::matmul_f32(A, B, &C).ok(), "matmul ok");
    expect(std::fabs(c[0] - 1.f) < 1e-5f, "matmul result");

    C.nbytes = 4;  // too small
    expect(!uaii::kernels::matmul_f32(A, B, &C).ok(), "matmul nbytes reject");
  }

  // Embedding OOB rejects
  {
    float ids[2] = {0.f, 99.f};
    float w[4] = {1, 2, 3, 4};
    float out[4] = {};
    std::int64_t tsh[2] = {1, 2};
    std::int64_t wsh[2] = {2, 2};
    std::int64_t osh[2] = {2, 2};
    uaii::kernels::TensorView T{uaii::DType::F32, tsh, 2, ids, sizeof(ids)};
    uaii::kernels::TensorView W{uaii::DType::F32, wsh, 2, w, sizeof(w)};
    uaii::kernels::TensorView O{uaii::DType::F32, osh, 2, out, sizeof(out)};
    expect(!uaii::kernels::embedding_f32(T, W, &O).ok(), "embedding OOB");
  }

  // Session toy MLP with explicit ones
  {
    uaii::ir::GraphBuilder b("rt_test");
    auto x = b.add_tensor("x", uaii::DType::F32, uaii::Shape{{1, 4}});
    auto w1 = b.add_weight("w1", uaii::DType::F32, uaii::Shape{{4, 4}}, "missing.bin");
    auto y = b.add_tensor("y", uaii::DType::F32, uaii::Shape{{1, 4}});
    b.add_node("m", "MatMul", "1", {x, w1}, {y});
    b.set_inputs({x}).set_outputs({y});

    uaii::runtime::SessionOptions bad;
    bad.weight_init = uaii::runtime::WeightInit::None;
    uaii::runtime::Session s1;
    expect(!s1.create(b.build(), bad).ok(), "fail closed missing weight");

    uaii::runtime::SessionOptions ok;
    ok.weight_init = uaii::runtime::WeightInit::Ones;
    uaii::runtime::Session s2;
    expect(s2.create(b.build(), ok).ok(), "ones weight init");
    expect(s2.set_tensor_f32("x", {1, 0, 0, 0}).ok(), "set x");
    expect(s2.run().ok(), "run");
  }

  // Validator MatMul shape
  {
    uaii::ir::GraphBuilder b("bad_mm");
    auto x = b.add_tensor("x", uaii::DType::F32, uaii::Shape{{1, 3}});
    auto w = b.add_weight("w", uaii::DType::F32, uaii::Shape{{4, 2}}, "w.bin");
    auto y = b.add_tensor("y", uaii::DType::F32, uaii::Shape{{1, 2}});
    b.add_node("m", "MatMul", "1", {x, w}, {y});
    b.set_inputs({x}).set_outputs({y});
    auto vr = uaii::ir::validate_graph(b.build(), uaii::ir::default_registry());
    expect(!vr.ok(), "validator catches matmul shape");
  }

  // Plugin op registry hot path
  {
    uaii::plugins::OperatorHostRegistry::instance().register_op(
        "TestDouble",
        [](const std::vector<uaii::kernels::TensorView>& in,
           std::vector<uaii::kernels::TensorView>* out,
           const std::vector<uaii::ir::Attribute>&) {
          if (in.size() != 1 || !out || out->size() != 1) {
            return uaii::Error::make(uaii::ErrorCode::InvalidArgument, "arity");
          }
          const float* x = in[0].f32();
          float* y = (*out)[0].f32();
          for (std::size_t i = 0; i < in[0].numel(); ++i) y[i] = x[i] * 2.f;
          return uaii::Error::success();
        });
    expect(uaii::kernels::supports_cpu_op("TestDouble", "1"), "plugin op supported");
  }

  // C API version + options defaults fail-closed
  {
    expect(std::string(uaii_get_c_api_version_string()) == "0.2.0", "c api 0.2.0");
    uaii_session_options opts;
    uaii_session_options_init(&opts);
    expect(opts.struct_size == sizeof(opts), "struct_size");
    expect(opts.weight_init == UAII_WEIGHT_INIT_NONE, "default weight none");
  }

  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "runtime_test OK\n";
  return 0;
}
