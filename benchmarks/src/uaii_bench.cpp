// UAII public microbenchmark harness (schema uaii_bench/v3).
//
// How to add a suite:
//  1. Implement bench_<name>(...) returning a report struct.
//  2. Add a SuiteId enum value + parse in parse_suites().
//  3. Call it from run_selected() and emit JSON/human sections.
//  4. Document the suite in docs/benchmarks.md.
//
// Design: absolute metrics first; multi-provider GEMM; --suite selects work.

#include "uaii/ir/graph.hpp"
#include "uaii/kernels/gemm.hpp"
#include "uaii/kernels/kernels.hpp"
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
#include <sstream>
#include <string>
#include <unordered_set>
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
using uaii::kernels::GemmProvider;
using uaii::kernels::IGemm;

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
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
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
  std::cout << "\"" << key << "\": {\"median_ms\": " << s.median_ms << ", \"min_ms\": " << s.min_ms
            << ", \"max_ms\": " << s.max_ms << ", \"trials\": " << s.trials << "}";
}

enum class SuiteId { Gemm, Bandwidth, Attention, Session, Quant };

bool parse_suite_token(const std::string& t, std::unordered_set<SuiteId>* out) {
  if (t == "all") {
    out->insert(
        {SuiteId::Gemm, SuiteId::Bandwidth, SuiteId::Attention, SuiteId::Session, SuiteId::Quant});
    return true;
  }
  if (t == "gemm") {
    out->insert(SuiteId::Gemm);
    return true;
  }
  if (t == "bandwidth") {
    out->insert(SuiteId::Bandwidth);
    return true;
  }
  if (t == "attention") {
    out->insert(SuiteId::Attention);
    return true;
  }
  if (t == "session") {
    out->insert(SuiteId::Session);
    return true;
  }
  if (t == "quant") {
    out->insert(SuiteId::Quant);
    return true;
  }
  return false;
}

std::unordered_set<SuiteId> parse_suites(const std::string& csv) {
  std::unordered_set<SuiteId> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
    while (!tok.empty() && tok.back() == ' ') tok.pop_back();
    if (tok.empty()) continue;
    if (!parse_suite_token(tok, &out)) {
      std::cerr << "unknown suite: " << tok << "\n";
    }
  }
  if (out.empty()) parse_suite_token("all", &out);
  return out;
}

GemmProvider parse_provider_token(const std::string& t) {
  if (t == "ref") return GemmProvider::Ref;
  if (t == "onednn") return GemmProvider::OneDnn;
  if (t == "openblas") return GemmProvider::OpenBlas;
  return GemmProvider::Ref;
}

std::vector<GemmProvider> parse_providers(const std::string& csv) {
  std::vector<GemmProvider> requested;
  if (csv.empty() || csv == "all") {
    return uaii::kernels::linked_gemm_providers();
  }
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
    while (!tok.empty() && tok.back() == ' ') tok.pop_back();
    if (tok.empty()) continue;
    const GemmProvider p = parse_provider_token(tok);
    if (!uaii::kernels::gemm_provider_linked(p)) {
      std::cerr << "provider not linked in this build (skipping): " << tok << "\n";
      continue;
    }
    if (std::find(requested.begin(), requested.end(), p) == requested.end()) {
      requested.push_back(p);
    }
  }
  if (requested.empty()) requested.push_back(GemmProvider::Ref);
  return requested;
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

struct SizeRow {
  std::int64_t n = 0;
  Stats ms;
  double gflops = 0;
  Stats naive;
  bool has_naive = false;
};

struct ProviderGemmReport {
  GemmProvider provider = GemmProvider::Ref;
  std::string name;
  std::vector<SizeRow> sizes;
};

SizeRow bench_gemm_size(IGemm& gemm, std::int64_t n, int warmup, int trials, bool vs_naive) {
  const std::size_t elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<float> a(elems), b(elems), c(elems), c_naive(elems);
  for (std::size_t i = 0; i < elems; ++i) {
    a[i] = 0.001f * static_cast<float>((i * 13) % 97);
    b[i] = 0.001f * static_cast<float>((i * 29) % 89);
  }
  SizeRow row;
  row.n = n;
  row.ms = time_trials(warmup, trials, [&]() {
    (void)gemm.gemm_f32(n, n, n, a.data(), n, false, b.data(), n, false, c.data(), n);
  });
  const double flops =
      2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
  row.gflops = (flops / (row.ms.median_ms / 1000.0)) / 1e9;
  if (vs_naive) {
    const int naive_trials = (n >= 1024) ? std::max(1, trials / 2) : trials;
    row.naive = time_trials(std::min(1, warmup), naive_trials, [&]() {
      naive_gemm_f32(n, a.data(), b.data(), c_naive.data());
    });
    row.has_naive = true;
  }
  return row;
}

std::vector<ProviderGemmReport> bench_gemm_suite(const std::vector<GemmProvider>& providers,
                                                 int warmup, int trials, bool vs_naive) {
  const std::vector<std::int64_t> sizes = {256, 512, 1024};
  std::vector<ProviderGemmReport> out;
  for (GemmProvider p : providers) {
    IGemm* g = uaii::kernels::try_get_gemm(p);
    if (!g) continue;
    uaii::kernels::GemmRegistry::instance().set_preferred(p);
    ProviderGemmReport rep;
    rep.provider = p;
    rep.name = g->name();
    for (std::int64_t n : sizes) {
      const int t = (n >= 1024) ? std::max(7, trials / 2) : trials;
      const bool do_naive = vs_naive && (n <= 512 || n == 1024);
      SizeRow row = bench_gemm_size(*g, n, warmup, t, do_naive && n <= 512);
      if (vs_naive && n == 1024) {
        row = bench_gemm_size(*g, n, 1, 3, true);
      }
      rep.sizes.push_back(row);
    }
    out.push_back(std::move(rep));
  }
  return out;
}

struct BandwidthKernel {
  std::string name;
  double bytes_per_iter = 0;
  Stats ms;
  double gbps = 0;
};

struct BandwidthReport {
  std::size_t elems = 0;
  double mib = 0;
  std::vector<BandwidthKernel> kernels;
};

BandwidthReport bench_bandwidth(int warmup, int trials) {
  // ~256 MiB working set across three arrays (STREAM-like).
  constexpr std::size_t kElems = (256ull * 1024ull * 1024ull) / (3ull * sizeof(float));
  std::vector<float> a(kElems), b(kElems), c(kElems);
  for (std::size_t i = 0; i < kElems; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
    c[i] = 3.0f;
  }
  const float scalar = 1.5f;
  BandwidthReport r;
  r.elems = kElems;
  r.mib = static_cast<double>(kElems * 3 * sizeof(float)) / (1024.0 * 1024.0);

  auto run_kernel = [&](const char* name, double bytes, const std::function<void()>& fn) {
    BandwidthKernel k;
    k.name = name;
    k.bytes_per_iter = bytes;
    k.ms = time_trials(warmup, trials, fn);
    k.gbps = (bytes / (k.ms.median_ms / 1000.0)) / 1e9;
    r.kernels.push_back(std::move(k));
  };

  const double b1 = static_cast<double>(kElems) * sizeof(float);
  run_kernel("copy", 2.0 * b1, [&]() {
    for (std::size_t i = 0; i < kElems; ++i) b[i] = a[i];
  });
  run_kernel("scale", 2.0 * b1, [&]() {
    for (std::size_t i = 0; i < kElems; ++i) b[i] = scalar * a[i];
  });
  run_kernel("add", 3.0 * b1, [&]() {
    for (std::size_t i = 0; i < kElems; ++i) c[i] = a[i] + b[i];
  });
  run_kernel("triad", 3.0 * b1, [&]() {
    for (std::size_t i = 0; i < kElems; ++i) a[i] = b[i] + scalar * c[i];
  });
  return r;
}

struct AttentionReport {
  std::int64_t batch = 1;
  std::int64_t heads = 8;
  std::int64_t seq = 512;
  std::int64_t head_dim = 64;
  std::string gemm_name;
  Stats e2e;
  double gemm_gflops = 0;
};

AttentionReport bench_attention(int warmup, int trials) {
  AttentionReport r;
  IGemm& gemm = uaii::kernels::default_gemm();
  r.gemm_name = gemm.name();
  const std::int64_t B = r.batch;
  const std::int64_t H = r.heads;
  const std::int64_t S = r.seq;
  const std::int64_t D = r.head_dim;
  const std::size_t qkv = static_cast<std::size_t>(B * H * S * D);
  const std::size_t scores_n = static_cast<std::size_t>(B * H * S * S);
  std::vector<float> q(qkv), k(qkv), v(qkv), scores(scores_n), probs(scores_n), out(qkv);
  for (std::size_t i = 0; i < qkv; ++i) {
    q[i] = 0.001f * static_cast<float>((i % 97) + 1);
    k[i] = 0.001f * static_cast<float>((i % 89) + 1);
    v[i] = 0.001f * static_cast<float>((i % 83) + 1);
  }

  // FLOPs: QK^T (2*S*S*D) + AV (2*S*S*D) per head * B * H
  r.gemm_gflops =
      (2.0 * static_cast<double>(B * H) *
       (2.0 * static_cast<double>(S) * static_cast<double>(S) * static_cast<double>(D))) /
      1e9;

  r.e2e = time_trials(warmup, trials, [&]() {
    for (std::int64_t b = 0; b < B; ++b) {
      for (std::int64_t h = 0; h < H; ++h) {
        const std::size_t off = static_cast<std::size_t>((b * H + h) * S * D);
        const std::size_t soff = static_cast<std::size_t>((b * H + h) * S * S);
        float* qq = q.data() + off;
        float* kk = k.data() + off;
        float* vv = v.data() + off;
        float* sc = scores.data() + soff;
        float* pr = probs.data() + soff;
        float* oo = out.data() + off;
        // scores = Q @ K^T  → [S,S]
        (void)gemm.gemm_f32(S, S, D, qq, D, false, kk, D, true, sc, S);
        std::int64_t shape[2] = {S, S};
        uaii::kernels::TensorView in;
        in.dtype = uaii::DType::F32;
        in.shape = shape;
        in.rank = 2;
        in.data = sc;
        in.nbytes = static_cast<std::size_t>(S * S) * sizeof(float);
        uaii::kernels::TensorView ou = in;
        ou.data = pr;
        (void)uaii::kernels::softmax_f32(in, &ou, -1);
        // out = probs @ V → [S,D]
        (void)gemm.gemm_f32(S, D, S, pr, S, false, vv, D, false, oo, D);
      }
    }
  });
  return r;
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
  r.q4_mib = static_cast<double>(quant::packed_nbytes(quant::QuantFormat::Q4_0, n)) /
             (1024.0 * 1024.0);
  r.mem_ratio_theoretical = (32.0 * 4.0) / 18.0;

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

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  bool vs_naive = false;
  int warmup = 5;
  int trials = 21;
  std::string suite_csv = "all";
  std::string providers_csv = "all";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") json = true;
    else if (a == "--vs-naive") vs_naive = true;
    else if (a == "--trials" && i + 1 < argc) trials = std::max(3, std::atoi(argv[++i]));
    else if (a == "--warmup" && i + 1 < argc) warmup = std::max(0, std::atoi(argv[++i]));
    else if (a == "--suite" && i + 1 < argc) suite_csv = argv[++i];
    else if (a == "--providers" && i + 1 < argc) providers_csv = argv[++i];
    else if (a == "--help" || a == "-h") {
      std::cout
          << "uaii_bench [options]\n"
          << "  --trials N              timed trials per metric (default 21, median)\n"
          << "  --warmup N              warmup runs discarded (default 5)\n"
          << "  --suite LIST            comma: all,gemm,bandwidth,attention,session,quant\n"
          << "  --providers LIST        comma: all,ref,onednn,openblas (linked only)\n"
          << "  --json                  machine-readable uaii_bench/v3 report\n"
          << "  --vs-naive              also time naive ijk GEMM (appendix; slow)\n";
      return 0;
    }
  }

  uaii::log::set_level(uaii::log::Level::Error);
  const auto suites = parse_suites(suite_csv);
  const auto providers = parse_providers(providers_csv);

  const std::string cpu = cpu_brand();
  const unsigned threads = uaii::kernels::hardware_concurrency();
#if defined(NDEBUG)
  const std::string build_type = "Release-ish (NDEBUG)";
#else
  const std::string build_type = "Debug-or-unoptimized (no NDEBUG)";
#endif
  const std::string scope =
      "Kernel microbenchmarks of UAII. Not LLM tokens/s. Not a bake-off vs llama.cpp/ORT/TRT.";

  std::vector<ProviderGemmReport> gemms;
  BandwidthReport bandwidth;
  bool have_bw = false;
  AttentionReport attention;
  bool have_attn = false;
  SessionRow session;
  bool have_session = false;
  QuantReport quant;
  bool have_quant = false;

  if (suites.count(SuiteId::Gemm)) {
    gemms = bench_gemm_suite(providers, warmup, trials, vs_naive);
  }
  if (suites.count(SuiteId::Bandwidth)) {
    bandwidth = bench_bandwidth(warmup, std::max(7, trials / 2));
    have_bw = true;
  }
  if (suites.count(SuiteId::Attention)) {
    // Use first requested provider for attention GEMMs.
    if (!providers.empty()) {
      uaii::kernels::GemmRegistry::instance().set_preferred(providers.front());
    }
    attention = bench_attention(warmup, std::max(7, trials / 2));
    have_attn = true;
  }
  if (suites.count(SuiteId::Session)) {
    session = bench_session_stack(warmup, trials);
    have_session = true;
  }
  if (suites.count(SuiteId::Quant)) {
    quant = bench_quant(2048, 4096, warmup, std::max(7, trials / 2));
    have_quant = true;
  }

  std::cout << std::fixed << std::setprecision(3);

  if (json) {
    std::cout << "{\n"
              << "  \"schema\": \"uaii_bench/v3\",\n"
              << "  \"uaii_version\": \"" << esc(uaii::version_string()) << "\",\n"
              << "  \"cpu\": \"" << esc(cpu) << "\",\n"
              << "  \"threads\": " << threads << ",\n"
              << "  \"build\": \"" << esc(build_type) << "\",\n"
              << "  \"linked_providers\": [";
    {
      const auto linked = uaii::kernels::linked_gemm_providers();
      for (std::size_t i = 0; i < linked.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << "\"" << uaii::kernels::to_string(linked[i]) << "\"";
      }
    }
    std::cout << "],\n"
              << "  \"scope\": \"" << esc(scope) << "\",\n"
              << "  \"warmup\": " << warmup << ",\n"
              << "  \"trials_default\": " << trials << ",\n";

    std::cout << "  \"gemm_by_provider\": [\n";
    for (std::size_t pi = 0; pi < gemms.size(); ++pi) {
      const auto& g = gemms[pi];
      std::cout << "    {\"provider\": \"" << uaii::kernels::to_string(g.provider)
                << "\", \"name\": \"" << esc(g.name) << "\", \"sizes\": [\n";
      for (std::size_t si = 0; si < g.sizes.size(); ++si) {
        const auto& s = g.sizes[si];
        std::cout << "      {\"n\": " << s.n << ", \"gflops_median\": " << s.gflops << ", ";
        print_stats_json("uaii_ms", s.ms);
        if (s.has_naive) {
          std::cout << ", ";
          print_stats_json("naive_ms", s.naive);
        }
        std::cout << "}";
        if (si + 1 < g.sizes.size()) std::cout << ",";
        std::cout << "\n";
      }
      std::cout << "    ]}";
      if (pi + 1 < gemms.size()) std::cout << ",";
      std::cout << "\n";
    }
    std::cout << "  ]";

    if (have_bw) {
      std::cout << ",\n  \"bandwidth\": {\"elems\": " << bandwidth.elems
                << ", \"working_set_mib\": " << bandwidth.mib << ", \"kernels\": [\n";
      for (std::size_t i = 0; i < bandwidth.kernels.size(); ++i) {
        const auto& k = bandwidth.kernels[i];
        std::cout << "    {\"name\": \"" << esc(k.name) << "\", \"bytes_per_iter\": "
                  << k.bytes_per_iter << ", \"gbps_median\": " << k.gbps << ", ";
        print_stats_json("ms", k.ms);
        std::cout << "}";
        if (i + 1 < bandwidth.kernels.size()) std::cout << ",";
        std::cout << "\n";
      }
      std::cout << "  ]}";
    }

    if (have_attn) {
      std::cout << ",\n  \"attention\": {\"batch\": " << attention.batch
                << ", \"heads\": " << attention.heads << ", \"seq\": " << attention.seq
                << ", \"head_dim\": " << attention.head_dim << ", \"gemm_name\": \""
                << esc(attention.gemm_name) << "\", \"gemm_gflops_nominal\": "
                << attention.gemm_gflops << ", ";
      print_stats_json("e2e_ms", attention.e2e);
      std::cout << "}";
    }

    if (have_session) {
      std::cout << ",\n  \"session\": {\"label\": \"" << esc(session.label)
                << "\", \"params\": " << session.params << ", \"nodes\": " << session.ops << ", ";
      print_stats_json("run_ms", session.run);
      std::cout << "}";
    }

    if (have_quant) {
      std::cout << ",\n  \"quant_q4_0\": {\"rows\": " << quant.rows << ", \"cols\": " << quant.cols
                << ", \"f32_mib\": " << quant.f32_mib << ", \"q4_mib\": " << quant.q4_mib
                << ", \"format_mem_ratio\": " << quant.mem_ratio_theoretical << ", ";
      print_stats_json("quant_gemm_ms", quant.quant);
      std::cout << ", ";
      print_stats_json("unpack_then_f32_ms", quant.unpack);
      std::cout << "}";
    }

    std::cout << "\n}\n";
    return 0;
  }

  // Human-readable
  std::cout << "# UAII microbenchmark report\n\n";
  std::cout << "**Scope:** " << scope << "\n\n";
  std::cout << "## Environment\n\n";
  std::cout << "| Field | Value |\n|---|---|\n";
  std::cout << "| UAII | " << uaii::version_string() << " |\n";
  std::cout << "| CPU | " << cpu << " |\n";
  std::cout << "| Threads | " << threads << " |\n";
  std::cout << "| Build | " << build_type << " |\n";
  std::cout << "| Linked GEMM | ";
  {
    const auto linked = uaii::kernels::linked_gemm_providers();
    for (std::size_t i = 0; i < linked.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << "`" << uaii::kernels::to_string(linked[i]) << "`";
    }
  }
  std::cout << " |\n";
  std::cout << "| Method | warmup=" << warmup << ", trials/median=" << trials << " |\n\n";

  if (!gemms.empty()) {
    std::cout << "## Dense f32 GEMM (by provider)\n\n";
    for (const auto& g : gemms) {
      std::cout << "### `" << uaii::kernels::to_string(g.provider) << "` (" << g.name << ")\n\n";
      std::cout << "| N | Median ms | Min–Max ms | GFLOP/s |\n|---:|---:|---:|---:|\n";
      for (const auto& s : g.sizes) {
        std::cout << "| " << s.n << " | " << s.ms.median_ms << " | " << s.ms.min_ms << "–"
                  << s.ms.max_ms << " | **" << s.gflops << "** |\n";
      }
      std::cout << "\n";
    }
  }

  if (have_bw) {
    std::cout << "## Memory bandwidth (STREAM-style)\n\n";
    std::cout << "Working set ≈ " << bandwidth.mib << " MiB (" << bandwidth.elems
              << " f32 elems × 3 arrays).\n\n";
    std::cout << "| Kernel | Median ms | GB/s |\n|---|---:|---:|\n";
    for (const auto& k : bandwidth.kernels) {
      std::cout << "| " << k.name << " | " << k.ms.median_ms << " | **" << k.gbps << "** |\n";
    }
    std::cout << "\n";
  }

  if (have_attn) {
    std::cout << "## Attention microbench\n\n";
    std::cout << "Shape: B=" << attention.batch << " H=" << attention.heads
              << " S=" << attention.seq << " D=" << attention.head_dim
              << " · GEMM=`" << attention.gemm_name << "`\n\n";
    std::cout << "| Metric | Value |\n|---|---:|\n";
    std::cout << "| Median e2e | **" << attention.e2e.median_ms << " ms** |\n";
    std::cout << "| Nominal GEMM FLOPs | " << attention.gemm_gflops << " GFLOP |\n\n";
  }

  if (have_session) {
    std::cout << "## Session graph (synthetic)\n\n";
    std::cout << "| " << session.label << " |\n\n";
    std::cout << "| Metric | Value |\n|---|---:|\n";
    std::cout << "| Parameters | " << session.params << " |\n";
    std::cout << "| Median `Session::run` | **" << session.run.median_ms << " ms** |\n\n";
  }

  if (have_quant) {
    std::cout << "## Q4_0 packed MatMul\n\n";
    std::cout << "| Metric | Value |\n|---|---:|\n";
    std::cout << "| Shape | " << quant.rows << "×" << quant.cols << " |\n";
    std::cout << "| f32 / Q4_0 | " << quant.f32_mib << " / **" << quant.q4_mib << "** MiB |\n";
    std::cout << "| Format compression | **" << quant.mem_ratio_theoretical << "×** |\n";
    std::cout << "| Packed quant-GEMM | **" << quant.quant.median_ms << " ms** |\n";
    std::cout << "| Unpack + f32 GEMM | " << quant.unpack.median_ms << " ms |\n\n";
  }

  return 0;
}
