#include "uaii/ir/graph.hpp"
#include "uaii/kernels/gemm.hpp"
#include "uaii/kernels/quant_gemm.hpp"
#include "uaii/kernels/tensor_view.hpp"
#include "uaii/quant/formats.hpp"
#include "uaii/quant/gguf_dequant.hpp"
#include "uaii/runtime/session.hpp"
#include "uaii/version.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using clock = std::chrono::steady_clock;

double ms_since(clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

void naive_gemm_f32(std::int64_t n, const float* a, const float* b, float* c) {
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float sum = 0.f;
      for (std::int64_t k = 0; k < n; ++k) {
        sum += a[i * n + k] * b[k * n + j];
      }
      c[i * n + j] = sum;
    }
  }
}

struct GemmResult {
  std::int64_t n = 0;
  double naive_ms = 0;
  double uaii_ms = 0;
  double gflops_uaii = 0;
  double speedup = 0;
};

GemmResult bench_gemm(std::int64_t n, int warmup, int iters) {
  const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<float> a(elems), b(elems), c_naive(elems), c_uaii(elems);
  for (std::size_t i = 0; i < elems; ++i) {
    a[i] = 0.001f * static_cast<float>((i * 13) % 97);
    b[i] = 0.001f * static_cast<float>((i * 29) % 89);
  }

  auto& gemm = uaii::kernels::default_gemm();
  for (int i = 0; i < warmup; ++i) {
    (void)gemm.gemm_f32(n, n, n, a.data(), n, false, b.data(), n, false, c_uaii.data(), n);
  }

  const int naive_iters = (n >= 1024) ? std::max(1, iters / 4) : iters;
  auto t0 = clock::now();
  for (int i = 0; i < naive_iters; ++i) {
    naive_gemm_f32(n, a.data(), b.data(), c_naive.data());
  }
  const double naive_ms = ms_since(t0) / naive_iters;

  t0 = clock::now();
  for (int i = 0; i < iters; ++i) {
    (void)gemm.gemm_f32(n, n, n, a.data(), n, false, b.data(), n, false, c_uaii.data(), n);
  }
  const double uaii_ms = ms_since(t0) / iters;

  const double flops =
      2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
  GemmResult r;
  r.n = n;
  r.naive_ms = naive_ms;
  r.uaii_ms = uaii_ms;
  r.gflops_uaii = (flops / (uaii_ms / 1000.0)) / 1e9;
  r.speedup = naive_ms > 0 ? (naive_ms / uaii_ms) : 0;
  return r;
}

double bench_session_ms(int warmup, int iters) {
  using namespace uaii;
  using namespace uaii::runtime;

  ir::GraphBuilder b("bench_mlp");
  const TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 4}});
  const TensorId w1 = b.add_weight("w1", DType::F32, Shape{{4, 64}}, "w1.bin");
  const TensorId h = b.add_tensor("h", DType::F32, Shape{{1, 64}});
  const TensorId ha = b.add_tensor("ha", DType::F32, Shape{{1, 64}});
  const TensorId w2 = b.add_weight("w2", DType::F32, Shape{{64, 4}}, "w2.bin");
  const TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 4}});
  const TensorId yp = b.add_tensor("yp", DType::F32, Shape{{1, 4}});
  b.add_node("fc1", "MatMul", "1", {x, w1}, {h});
  b.add_node("act", "Relu", "1", {h}, {ha});
  b.add_node("fc2", "MatMul", "1", {ha, w2}, {y});
  b.add_node("sm", "Softmax", "1", {y}, {yp}, {ir::make_int_attr("axis", -1)});
  b.set_inputs({x}).set_outputs({yp});

  SessionOptions opts;
  opts.weight_init = WeightInit::Ones;
  opts.enable_fusion = true;
  opts.enable_memory_reuse = true;
  Session session;
  Error err = session.create(b.build(), opts);
  if (!err.ok()) return -1.0;
  err = session.set_tensor_f32("x", {1.f, 2.f, 3.f, 4.f});
  if (!err.ok()) return -1.0;

  for (int i = 0; i < warmup; ++i) {
    (void)session.run();
  }
  auto t0 = clock::now();
  for (int i = 0; i < iters; ++i) {
    err = session.run();
    if (!err.ok()) return -1.0;
  }
  return ms_since(t0) / iters;
}

struct QuantResult {
  double unpack_gemm_ms = 0;
  double quant_gemm_ms = 0;
  double speedup = 0;
  double f32_weight_mib = 0;
  double q4_weight_mib = 0;
  double mem_ratio = 0;
};

QuantResult bench_quant(std::int64_t rows, std::int64_t cols, int warmup, int iters) {
  using namespace uaii;
  QuantResult r;
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  r.f32_weight_mib = static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0);
  r.q4_weight_mib =
      static_cast<double>(quant::packed_nbytes(quant::QuantFormat::Q4_0, n)) / (1024.0 * 1024.0);
  r.mem_ratio = r.q4_weight_mib > 0 ? (r.f32_weight_mib / r.q4_weight_mib) : 0;

  const std::size_t packed_n = quant::packed_nbytes(quant::QuantFormat::Q4_0, n);
  std::vector<std::uint8_t> packed(packed_n, 0);
  const std::int64_t blocks = static_cast<std::int64_t>(n / 32);
  for (std::int64_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = packed.data() + static_cast<std::size_t>(b) * 18;
    blk[0] = 0x00;
    blk[1] = 0x3c;  // f16 1.0
    for (int i = 0; i < 16; ++i) {
      blk[2 + i] = static_cast<std::uint8_t>((i + static_cast<int>(b)) & 0xff);
    }
  }

  std::vector<float> a(static_cast<std::size_t>(cols));
  for (std::int64_t i = 0; i < cols; ++i) {
    a[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>(i % 17);
  }
  std::vector<float> c_q(static_cast<std::size_t>(rows), 0.f);
  std::vector<float> c_u(static_cast<std::size_t>(rows), 0.f);
  std::vector<float> w_f32(n, 0.f);

  const std::int64_t a_shape[2] = {1, cols};
  const std::int64_t b_shape[2] = {rows, cols};
  const std::int64_t c_shape[2] = {1, rows};

  kernels::TensorView av;
  av.data = a.data();
  av.dtype = DType::F32;
  av.rank = 2;
  av.shape = a_shape;
  av.nbytes = a.size() * sizeof(float);

  kernels::TensorView bq;
  bq.data = packed.data();
  bq.dtype = DType::Unknown;
  bq.quant_format = quant::QuantFormat::Q4_0;
  bq.quant_rows = rows;
  bq.quant_cols = cols;
  bq.rank = 2;
  bq.shape = b_shape;
  bq.nbytes = packed_n;

  kernels::TensorView cq;
  cq.data = c_q.data();
  cq.dtype = DType::F32;
  cq.rank = 2;
  cq.shape = c_shape;
  cq.nbytes = c_q.size() * sizeof(float);

  for (int i = 0; i < warmup; ++i) {
    (void)kernels::quant_gemm_f32(av, bq, &cq, true);
  }
  auto t0 = clock::now();
  for (int i = 0; i < iters; ++i) {
    (void)kernels::quant_gemm_f32(av, bq, &cq, true);
  }
  r.quant_gemm_ms = ms_since(t0) / iters;

  auto& gemm = kernels::default_gemm();
  const std::size_t row_bytes = quant::packed_nbytes(quant::QuantFormat::Q4_0,
                                                     static_cast<std::size_t>(cols));
  auto unpack_all = [&]() {
    for (std::int64_t row = 0; row < rows; ++row) {
      const std::uint8_t* prow = packed.data() + row_bytes * static_cast<std::size_t>(row);
      (void)quant::dequant_gguf_row(quant::QuantFormat::Q4_0, prow, cols,
                                    w_f32.data() + static_cast<std::size_t>(row) * cols);
    }
  };
  for (int i = 0; i < warmup; ++i) {
    unpack_all();
    (void)gemm.gemm_f32(1, rows, cols, a.data(), cols, false, w_f32.data(), cols, true,
                        c_u.data(), rows);
  }
  t0 = clock::now();
  for (int i = 0; i < iters; ++i) {
    unpack_all();
    (void)gemm.gemm_f32(1, rows, cols, a.data(), cols, false, w_f32.data(), cols, true,
                        c_u.data(), rows);
  }
  r.unpack_gemm_ms = ms_since(t0) / iters;
  r.speedup = r.quant_gemm_ms > 0 ? (r.unpack_gemm_ms / r.quant_gemm_ms) : 0;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  int warmup = 2;
  int iters = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--iters" && i + 1 < argc) {
      iters = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--warmup" && i + 1 < argc) {
      warmup = std::max(0, std::atoi(argv[++i]));
    } else if (a == "--help" || a == "-h") {
      std::cout << "uaii_bench [--iters N] [--warmup N] [--json]\n";
      return 0;
    }
  }

  const auto gemm512 = bench_gemm(512, warmup, iters);
  const auto gemm1024 = bench_gemm(1024, warmup, std::max(3, iters / 2));
  const double session_ms = bench_session_ms(warmup, iters * 5);
  const auto quant = bench_quant(1024, 4096, warmup, iters);

  const char* gemm_name = uaii::kernels::default_gemm().name();
  const std::string provider = uaii::kernels::GemmRegistry::instance().describe();

  if (json) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\n"
              << "  \"uaii_version\": \"" << uaii::version_string() << "\",\n"
              << "  \"gemm_provider\": \"" << provider << "\",\n"
              << "  \"gemm_name\": \"" << gemm_name << "\",\n"
              << "  \"gemm_512\": {\"naive_ms\": " << gemm512.naive_ms
              << ", \"uaii_ms\": " << gemm512.uaii_ms << ", \"gflops\": " << gemm512.gflops_uaii
              << ", \"speedup_x\": " << gemm512.speedup << "},\n"
              << "  \"gemm_1024\": {\"naive_ms\": " << gemm1024.naive_ms
              << ", \"uaii_ms\": " << gemm1024.uaii_ms << ", \"gflops\": " << gemm1024.gflops_uaii
              << ", \"speedup_x\": " << gemm1024.speedup << "},\n"
              << "  \"session_mlp_ms\": " << session_ms << ",\n"
              << "  \"quant_q4_0\": {\"rows\": 1024, \"cols\": 4096, \"unpack_gemm_ms\": "
              << quant.unpack_gemm_ms << ", \"quant_gemm_ms\": " << quant.quant_gemm_ms
              << ", \"speedup_x\": " << quant.speedup << ", \"f32_mib\": " << quant.f32_weight_mib
              << ", \"q4_mib\": " << quant.q4_weight_mib << ", \"mem_ratio\": " << quant.mem_ratio
              << "}\n"
              << "}\n";
    return 0;
  }

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "UAII microbench " << uaii::version_string() << "\n";
  std::cout << "GEMM provider: " << provider << " (" << gemm_name << ")\n";
  std::cout << "iters=" << iters << " warmup=" << warmup << "\n\n";

  std::cout << "## CPU GEMM (f32, square)\n";
  std::cout << "| Size | Naive | UAII | Speedup | GFLOP/s |\n";
  std::cout << "|---|---:|---:|---:|---:|\n";
  std::cout << "| 512³ | " << gemm512.naive_ms << " ms | " << gemm512.uaii_ms << " ms | "
            << gemm512.speedup << "× | " << gemm512.gflops_uaii << " |\n";
  std::cout << "| 1024³ | " << gemm1024.naive_ms << " ms | " << gemm1024.uaii_ms << " ms | "
            << gemm1024.speedup << "× | " << gemm1024.gflops_uaii << " |\n\n";

  std::cout << "## Session (fused MLP)\n";
  std::cout << "| Metric | Result |\n|---|---:|\n";
  std::cout << "| Avg run | " << session_ms << " ms |\n\n";

  std::cout << "## Q4_0 in-memory GEMM (1×1024 @ 1024×4096)\n";
  std::cout << "| Path | Time | Notes |\n|---|---:|---|\n";
  std::cout << "| Unpack + f32 GEMM | " << quant.unpack_gemm_ms << " ms | full dequant |\n";
  std::cout << "| Quant GEMM | " << quant.quant_gemm_ms << " ms | **" << quant.speedup
            << "×** |\n";
  std::cout << "| Weight memory | " << quant.q4_weight_mib << " MiB | vs " << quant.f32_weight_mib
            << " MiB f32 (**" << quant.mem_ratio << "×** smaller) |\n";
  return 0;
}
