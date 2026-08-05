// UAII public microbenchmark harness.
// Design goals for external credibility:
//  - Absolute metrics first (GFLOP/s, ms), not "Nx vs naive"
//  - Multi-trial median + min/max
//  - Full environment disclosure in every report
//  - Scope clearly labeled (kernel microbench ≠ LLM tokens/s)
//  - Optional --vs-naive appendix only (never the headline)

#include "uaii/ir/graph.hpp"
#include "uaii/kernels/gemm.hpp"
#include "uaii/kernels/quant_gemm.hpp"
#include "uaii/kernels/tensor_view.hpp"
#include "uaii/kernels/thread_pool.hpp"
#include "uaii/quant/formats.hpp"
#include "uaii/quant/gguf_dequant.hpp"
#include "uaii/core/log.hpp"
#include "uaii/runtime/session.hpp"
#include "uaii/version.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
#  if defined(_WIN32) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
      defined(_M_IX86)
#    define UAII_BENCH_HAVE_X86_CPUID 1
#  endif
#endif

#if defined(UAII_BENCH_HAVE_X86_CPUID)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

namespace {

using clock = std::chrono::steady_clock;

double ms_since(clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

struct Stats {
  double median_ms = 0;
  double min_ms = 0;
  double max_ms = 0;
  int trials = 0;
};

Stats median_stats(std::vector<double> samples) {
  Stats s;
  s.trials = static_cast<int>(samples.size());
  if (samples.empty()) return s;
  std::sort(samples.begin(), samples.end());
  s.min_ms = samples.front();
  s.max_ms = samples.back();
  const std::size_t mid = samples.size() / 2;
  if (samples.size() % 2 == 1) {
    s.median_ms = samples[mid];
  } else {
    s.median_ms = 0.5 * (samples[mid - 1] + samples[mid]);
  }
  return s;
}

Stats time_trials(int warmup, int trials, const std::function<void()>& fn) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(trials));
  for (int i = 0; i < trials; ++i) {
    const auto t0 = clock::now();
    fn();
    samples.push_back(ms_since(t0));
  }
  return median_stats(std::move(samples));
}

std::string cpu_brand() {
  if (const char* env = std::getenv("UAII_BENCH_CPU")) {
    if (env[0] != '\0') return env;
  }
#if defined(UAII_BENCH_HAVE_X86_CPUID)
  char brand[49] = {};
#  if defined(_MSC_VER)
  int regs[4] = {};
  __cpuid(regs, 0x80000000);
  const unsigned max_ext = static_cast<unsigned>(regs[0]);
  if (max_ext < 0x80000004u) return "unknown-cpu";
  __cpuid(regs, 0x80000002);
  std::memcpy(brand + 0, regs, 16);
  __cpuid(regs, 0x80000003);
  std::memcpy(brand + 16, regs, 16);
  __cpuid(regs, 0x80000004);
  std::memcpy(brand + 32, regs, 16);
#  else
  unsigned max_ext = 0;
  if (__get_cpuid_max(0x80000000u, &max_ext) == 0 || max_ext < 0x80000004u) {
    return "unknown-cpu";
  }
  unsigned eax, ebx, ecx, edx;
  __get_cpuid(0x80000002, &eax, &ebx, &ecx, &edx);
  std::memcpy(brand + 0, &eax, 4);
  std::memcpy(brand + 4, &ebx, 4);
  std::memcpy(brand + 8, &ecx, 4);
  std::memcpy(brand + 12, &edx, 4);
  __get_cpuid(0x80000003, &eax, &ebx, &ecx, &edx);
  std::memcpy(brand + 16, &eax, 4);
  std::memcpy(brand + 20, &ebx, 4);
  std::memcpy(brand + 24, &ecx, 4);
  std::memcpy(brand + 28, &edx, 4);
  __get_cpuid(0x80000004, &eax, &ebx, &ecx, &edx);
  std::memcpy(brand + 32, &eax, 4);
  std::memcpy(brand + 36, &ebx, 4);
  std::memcpy(brand + 40, &ecx, 4);
  std::memcpy(brand + 44, &edx, 4);
#  endif
  brand[48] = '\0';
  std::string s(brand);
  const auto start = s.find_first_not_of(' ');
  if (start == std::string::npos) return "unknown-cpu";
  const auto end = s.find_last_not_of(' ');
  return s.substr(start, end - start + 1);
#else
  return "unknown-cpu";
#endif
}

struct EnvInfo {
  std::string uaii_version;
  std::string gemm_provider;
  std::string gemm_name;
  std::string cpu;
  unsigned threads = 0;
  std::string build_type;
  std::string note;
};

EnvInfo capture_env() {
  EnvInfo e;
  e.uaii_version = uaii::version_string();
  e.gemm_provider = uaii::kernels::GemmRegistry::instance().describe();
  e.gemm_name = uaii::kernels::default_gemm().name();
  e.cpu = cpu_brand();
  if (const char* override_cpu = std::getenv("UAII_BENCH_CPU")) {
    if (override_cpu[0] != '\0') e.cpu = override_cpu;
  }
  e.threads = uaii::kernels::hardware_concurrency();
#if defined(NDEBUG)
  e.build_type = "Release-ish (NDEBUG)";
#else
  e.build_type = "Debug-or-unoptimized (no NDEBUG)";
#endif
  e.note =
      "Kernel microbenchmarks of UAII. Not LLM tokens/s. Not a bake-off vs llama.cpp/ORT/TRT.";
  return e;
}

void naive_gemm_f32(std::int64_t n, const float* a, const float* b, float* c) {
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float sum = 0.f;
      for (std::int64_t k = 0; k < n; ++k) sum += a[i * n + k] * b[k * n + j];
      c[i * n + j] = sum;
    }
  }
}

struct GemmRow {
  std::int64_t n = 0;
  Stats uaii;
  double gflops = 0;
  Stats naive;  // only if requested
  bool has_naive = false;
};

GemmRow bench_gemm(std::int64_t n, int warmup, int trials, bool vs_naive) {
  const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<float> a(elems), b(elems), c(elems), c_naive(elems);
  for (std::size_t i = 0; i < elems; ++i) {
    a[i] = 0.001f * static_cast<float>((i * 13) % 97);
    b[i] = 0.001f * static_cast<float>((i * 29) % 89);
  }
  auto& gemm = uaii::kernels::default_gemm();

  GemmRow row;
  row.n = n;
  row.uaii = time_trials(warmup, trials, [&]() {
    (void)gemm.gemm_f32(n, n, n, a.data(), n, false, b.data(), n, false, c.data(), n);
  });
  const double flops =
      2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
  row.gflops = (flops / (row.uaii.median_ms / 1000.0)) / 1e9;

  if (vs_naive) {
    // Fewer trials for large naive GEMM (very slow).
    const int naive_trials = (n >= 1024) ? std::max(1, trials / 2) : trials;
    row.naive = time_trials(std::min(1, warmup), naive_trials, [&]() {
      naive_gemm_f32(n, a.data(), b.data(), c_naive.data());
    });
    row.has_naive = true;
  }
  return row;
}

struct SessionRow {
  std::string label;
  Stats run;
  std::int64_t ops = 0;
  std::int64_t params = 0;
};

SessionRow bench_session_stack(int warmup, int trials) {
  using namespace uaii;
  using namespace uaii::runtime;

  // Synthetic stack: more representative than a 4-wide toy, still not an LLM.
  constexpr std::int64_t kDim = 512;
  constexpr std::int64_t kLayers = 8;
  ir::GraphBuilder b("bench_mlp_stack");
  TensorId x = b.add_tensor("x", DType::F32, Shape{{1, kDim}});
  TensorId cur = x;
  std::int64_t params = 0;
  for (std::int64_t li = 0; li < kLayers; ++li) {
    const std::string p = "L" + std::to_string(li) + ".";
    TensorId w = b.add_weight(p + "w", DType::F32, Shape{{kDim, kDim}}, p + "w.bin");
    TensorId y = b.add_tensor(p + "y", DType::F32, Shape{{1, kDim}});
    TensorId a = b.add_tensor(p + "a", DType::F32, Shape{{1, kDim}});
    b.add_node(p + "mm", "MatMul", "1", {cur, w}, {y});
    b.add_node(p + "act", "Relu", "1", {y}, {a});
    cur = a;
    params += kDim * kDim;
  }
  TensorId out = b.add_tensor("out", DType::F32, Shape{{1, kDim}});
  b.add_node("sm", "Softmax", "1", {cur}, {out}, {ir::make_int_attr("axis", -1)});
  b.set_inputs({x}).set_outputs({out});

  SessionOptions opts;
  opts.weight_init = WeightInit::Ones;
  opts.enable_fusion = true;
  opts.enable_memory_reuse = true;
  opts.enable_profiler = false;
  Session session;
  Error err = session.create(b.build(), opts);
  SessionRow row;
  row.label = "8× MatMul(512²)+ReLU + Softmax (synthetic IR)";
  row.ops = kLayers * 2 + 1;
  row.params = params;
  if (!err.ok()) {
    row.run.median_ms = -1;
    return row;
  }
  std::vector<float> xin(static_cast<std::size_t>(kDim), 0.01f);
  err = session.set_tensor_f32("x", xin);
  if (!err.ok()) {
    row.run.median_ms = -1;
    return row;
  }

  row.run = time_trials(warmup, trials, [&]() { (void)session.run(); });
  return row;
}

struct QuantReport {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  double f32_mib = 0;
  double q4_mib = 0;
  double mem_ratio_theoretical = 0;
  Stats quant;
  Stats unpack;
};

QuantReport bench_quant(std::int64_t rows, std::int64_t cols, int warmup, int trials) {
  using namespace uaii;
  QuantReport r;
  r.rows = rows;
  r.cols = cols;
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  r.f32_mib = static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0);
  // GGUF Q4_0: 18 bytes / 32 values → 18/128 of f32 footprint.
  r.q4_mib = static_cast<double>(quant::packed_nbytes(quant::QuantFormat::Q4_0, n)) /
             (1024.0 * 1024.0);
  r.mem_ratio_theoretical = (32.0 * 4.0) / 18.0;  // exact format ratio

  const std::size_t packed_n = quant::packed_nbytes(quant::QuantFormat::Q4_0, n);
  std::vector<std::uint8_t> packed(packed_n, 0);
  const std::int64_t blocks = static_cast<std::int64_t>(n / 32);
  for (std::int64_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = packed.data() + static_cast<std::size_t>(b) * 18;
    blk[0] = 0x00;
    blk[1] = 0x3c;
    for (int i = 0; i < 16; ++i) blk[2 + i] = static_cast<std::uint8_t>((i + b) & 0xff);
  }

  std::vector<float> a(static_cast<std::size_t>(cols));
  for (std::int64_t i = 0; i < cols; ++i) {
    a[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>((i % 17) + 1);
  }
  std::vector<float> c_q(static_cast<std::size_t>(rows), 0.f);
  std::vector<float> c_u(static_cast<std::size_t>(rows), 0.f);
  std::vector<float> w_f32(n, 0.f);

  const std::int64_t a_shape[2] = {1, cols};
  const std::int64_t b_shape[2] = {rows, cols};
  const std::int64_t c_shape[2] = {1, rows};

  kernels::TensorView av;
  av.dtype = DType::F32;
  av.shape = a_shape;
  av.rank = 2;
  av.data = a.data();
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
  cq.dtype = DType::F32;
  cq.shape = c_shape;
  cq.rank = 2;
  cq.data = c_q.data();
  cq.nbytes = c_q.size() * sizeof(float);

  r.quant = time_trials(warmup, trials, [&]() {
    (void)kernels::quant_gemm_f32(av, bq, &cq, true);
  });

  auto& gemm = kernels::default_gemm();
  const std::size_t row_bytes =
      quant::packed_nbytes(quant::QuantFormat::Q4_0, static_cast<std::size_t>(cols));
  r.unpack = time_trials(warmup, trials, [&]() {
    for (std::int64_t row = 0; row < rows; ++row) {
      const std::uint8_t* prow = packed.data() + row_bytes * static_cast<std::size_t>(row);
      (void)quant::dequant_gguf_row(quant::QuantFormat::Q4_0, prow, cols,
                                    w_f32.data() + static_cast<std::size_t>(row) * cols);
    }
    (void)gemm.gemm_f32(1, rows, cols, a.data(), cols, false, w_f32.data(), cols, true,
                        c_u.data(), rows);
  });
  return r;
}

std::string esc(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (char c : s) {
    if (c == '"' || c == '\\') o.push_back('\\');
    o.push_back(c);
  }
  return o;
}

void print_stats_json(const char* key, const Stats& s) {
  std::cout << "    \"" << key << "\": {\"median_ms\": " << s.median_ms << ", \"min_ms\": "
            << s.min_ms << ", \"max_ms\": " << s.max_ms << ", \"trials\": " << s.trials << "}";
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  bool vs_naive = false;
  int warmup = 5;
  int trials = 21;  // odd → clean median
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") json = true;
    else if (a == "--vs-naive") vs_naive = true;
    else if (a == "--trials" && i + 1 < argc) trials = std::max(3, std::atoi(argv[++i]));
    else if (a == "--warmup" && i + 1 < argc) warmup = std::max(0, std::atoi(argv[++i]));
    else if (a == "--help" || a == "-h") {
      std::cout
          << "uaii_bench [options]\n"
          << "  --trials N     timed trials per metric (default 21, median reported)\n"
          << "  --warmup N     warmup runs discarded (default 5)\n"
          << "  --json         machine-readable report\n"
          << "  --vs-naive     also time naive ijk GEMM (appendix only; slow)\n";
      return 0;
    }
  }

  // Quiet session logs during bench.
  uaii::log::set_level(uaii::log::Level::Error);

  const EnvInfo env = capture_env();
  const std::vector<std::int64_t> sizes = {256, 512, 1024};
  std::vector<GemmRow> gemms;
  gemms.reserve(sizes.size());
  for (std::int64_t n : sizes) {
    // Large sizes: fewer trials still OK for median if trials>=7
    const int t = (n >= 1024) ? std::max(7, trials / 2) : trials;
    gemms.push_back(bench_gemm(n, warmup, t, vs_naive && n <= 512));
  }
  // Optional naive only at 1024 if explicitly requested (very slow).
  if (vs_naive) {
    for (auto& g : gemms) {
      if (g.n == 1024 && !g.has_naive) {
        g = bench_gemm(1024, 1, 3, true);
      }
    }
  }

  const SessionRow session = bench_session_stack(warmup, trials);
  const QuantReport quant = bench_quant(2048, 4096, warmup, std::max(7, trials / 2));

  std::cout << std::fixed << std::setprecision(3);

  if (json) {
    std::cout << "{\n"
              << "  \"schema\": \"uaii_bench/v2\",\n"
              << "  \"uaii_version\": \"" << esc(env.uaii_version) << "\",\n"
              << "  \"cpu\": \"" << esc(env.cpu) << "\",\n"
              << "  \"threads\": " << env.threads << ",\n"
              << "  \"build\": \"" << esc(env.build_type) << "\",\n"
              << "  \"gemm_provider\": \"" << esc(env.gemm_provider) << "\",\n"
              << "  \"gemm_name\": \"" << esc(env.gemm_name) << "\",\n"
              << "  \"scope\": \"" << esc(env.note) << "\",\n"
              << "  \"warmup\": " << warmup << ",\n"
              << "  \"trials_default\": " << trials << ",\n"
              << "  \"gemm\": [\n";
    for (std::size_t i = 0; i < gemms.size(); ++i) {
      const auto& g = gemms[i];
      std::cout << "    {\"n\": " << g.n << ", \"gflops_median\": " << g.gflops << ",\n";
      print_stats_json("uaii_ms", g.uaii);
      if (g.has_naive) {
        std::cout << ",\n";
        print_stats_json("naive_ms", g.naive);
      }
      std::cout << "}";
      if (i + 1 < gemms.size()) std::cout << ",";
      std::cout << "\n";
    }
    std::cout << "  ],\n"
              << "  \"session\": {\"label\": \"" << esc(session.label)
              << "\", \"params\": " << session.params << ", \"nodes\": " << session.ops << ",\n";
    print_stats_json("run_ms", session.run);
    std::cout << "},\n"
              << "  \"quant_q4_0\": {\"rows\": " << quant.rows << ", \"cols\": " << quant.cols
              << ", \"f32_mib\": " << quant.f32_mib << ", \"q4_mib\": " << quant.q4_mib
              << ", \"format_mem_ratio\": " << quant.mem_ratio_theoretical << ",\n";
    print_stats_json("quant_gemm_ms", quant.quant);
    std::cout << ",\n";
    print_stats_json("unpack_then_f32_ms", quant.unpack);
    std::cout << "}\n}\n";
    return 0;
  }

  std::cout << "# UAII microbenchmark report\n\n";
  std::cout << "**Scope:** " << env.note << "\n\n";
  std::cout << "## Environment\n\n";
  std::cout << "| Field | Value |\n|---|---|\n";
  std::cout << "| UAII | " << env.uaii_version << " |\n";
  std::cout << "| CPU | " << env.cpu << " |\n";
  std::cout << "| Threads (`UAII_NUM_THREADS` / hw) | " << env.threads << " |\n";
  std::cout << "| GEMM provider | `" << env.gemm_provider << "` |\n";
  std::cout << "| Build | " << env.build_type << " |\n";
  std::cout << "| Method | warmup=" << warmup << ", trials/median (default " << trials
            << ") |\n\n";

  std::cout << "## 1. Dense f32 GEMM (absolute)\n\n";
  std::cout << "Square `C = A @ B`, row-major. Throughput from **median** trial time.\n\n";
  std::cout << "| N | Median ms | Min–Max ms | GFLOP/s (median) |\n";
  std::cout << "|---:|---:|---:|---:|\n";
  for (const auto& g : gemms) {
    std::cout << "| " << g.n << " | " << g.uaii.median_ms << " | " << g.uaii.min_ms << "–"
              << g.uaii.max_ms << " | **" << g.gflops << "** |\n";
  }
  std::cout << "\n";

  if (vs_naive) {
    std::cout << "### Appendix — vs naive triple-loop (not a product claim)\n\n";
    std::cout << "Naive `ijk` has no tiling/threading; included only for engineering sanity.\n\n";
    std::cout << "| N | Naive median ms | UAII median ms | Ratio |\n|---:|---:|---:|---:|\n";
    for (const auto& g : gemms) {
      if (!g.has_naive) continue;
      const double ratio =
          g.uaii.median_ms > 0 ? (g.naive.median_ms / g.uaii.median_ms) : 0.0;
      std::cout << "| " << g.n << " | " << g.naive.median_ms << " | " << g.uaii.median_ms
                << " | " << ratio << "× |\n";
    }
    std::cout << "\n";
  }

  std::cout << "## 2. Session graph (synthetic)\n\n";
  std::cout << "|" << session.label << "|\n\n";
  std::cout << "| Metric | Value |\n|---|---:|\n";
  std::cout << "| Parameters (f32 weights) | " << session.params << " |\n";
  std::cout << "| Median `Session::run` | **" << session.run.median_ms << " ms** |\n";
  std::cout << "| Min–Max | " << session.run.min_ms << "–" << session.run.max_ms << " ms |\n\n";

  std::cout << "## 3. Q4_0 packed MatMul\n\n";
  std::cout << "Shape: activation `[1," << quant.cols << "]` × weight `[" << quant.rows << ","
            << quant.cols << "]` (GGUF Q4_0 blocks). "
            << "Memory ratio is **format-defined** (32×f32 / 18 B = "
            << quant.mem_ratio_theoretical << "×).\n\n";
  std::cout << "| Metric | Value |\n|---|---:|\n";
  std::cout << "| f32 weight bytes | " << quant.f32_mib << " MiB |\n";
  std::cout << "| Q4_0 packed bytes | **" << quant.q4_mib << " MiB** |\n";
  std::cout << "| Format compression | **" << quant.mem_ratio_theoretical << "×** |\n";
  std::cout << "| Packed quant-GEMM median | **" << quant.quant.median_ms << " ms** |\n";
  std::cout << "| Unpack-all + f32 GEMM median | " << quant.unpack.median_ms << " ms |\n";
  if (quant.quant.median_ms > 0) {
    std::cout << "| Time ratio (unpack path / packed) | "
              << (quant.unpack.median_ms / quant.quant.median_ms) << "× |\n";
  }
  std::cout << "\n*Synthetic Q4_0 payload (valid block layout). Not a full GGUF model.*\n";
  return 0;
}
