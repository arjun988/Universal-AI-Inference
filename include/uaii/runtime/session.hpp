#pragma once

#include "uaii/backends/parity.hpp"
#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/backend.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/plan.hpp"
#include "uaii/kernels/tensor_view.hpp"
#include "uaii/memory/allocator.hpp"
#include "uaii/memory/buffer.hpp"
#include "uaii/planner/optimize.hpp"
#include "uaii/profiler/profiler.hpp"
#include "uaii/quant/formats.hpp"
#include "uaii/runtime/kv_cache.hpp"
#include "uaii/runtime/sampling.hpp"
#include "uaii/runtime/scheduler_cpu.hpp"
#include "uaii/runtime/scheduler_device.hpp"
#include "uaii/storage/streaming.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace runtime {

enum class WeightInit {
  None = 0,
  Zeros,
  Ones,
  Sequence,
};

struct SessionOptions {
  memory::AllocatorConfig allocator;
  WeightInit weight_init = WeightInit::None;
  std::string weights_dir;
  /// If false (default), a failed weight_ref load is an error (no silent Ones fill).
  bool allow_missing_weights = false;
  /// If set, weight paths must resolve under this directory (path sandbox).
  std::string weights_sandbox;
  bool validate = true;
  bool allow_unknown_ops = false;
  /// Backend name: cpu, cuda, metal, vulkan, webgpu, rocm.
  std::string backend_name = "cpu";
  bool prefer_native = true;
  bool force_host_fallback = false;

  // Phase 6
  bool enable_fusion = true;
  bool enable_memory_reuse = true;
  bool enable_storage_plan = true;
  bool enable_plan_cache = true;
  bool enable_streaming = false;
  bool enable_profiler = false;
  std::string profile_trace_path;
  quant::QuantFormat weight_quant = quant::QuantFormat::F32;
  /// Activation compute dtype: F32 (default) or F16 (stored/computed as f32 today with policy flag).
  DType compute_dtype = DType::F32;
  /// Keep GGUF block-quant weights packed for in-memory quant GEMM when true.
  bool keep_quantized_weights = true;
  /// Max context length for generate / KV (0 = from graph metadata or unlimited).
  std::int64_t max_context = 0;
};

struct OptimizeReport {
  std::string summary;
  planner::FusionStats fusion;
  planner::MemoryReusePlan memory;
  planner::StoragePlan storage;
  bool cache_hit = false;
};

/// End-to-end inference session over a UAII IR graph (any registered backend).
class UAII_API Session {
 public:
  Session();
  ~Session();

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] Error create(ir::Graph graph, SessionOptions options = {});
  void destroy() noexcept;

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] const ir::Graph& graph() const noexcept { return graph_; }
  [[nodiscard]] const ir::ExecutionPlan& plan() const noexcept { return plan_; }
  [[nodiscard]] const memory::Allocator& allocator() const noexcept { return *allocator_; }
  [[nodiscard]] const IBackend* backend() const noexcept { return backend_.get(); }
  [[nodiscard]] const OptimizeReport& optimize_report() const noexcept { return report_; }
  [[nodiscard]] const profiler::Profiler& profiler() const noexcept { return profiler_; }
  [[nodiscard]] profiler::Profiler& profiler() noexcept { return profiler_; }

  [[nodiscard]] Error set_tensor(const std::string& name_or_id,
                                 const void* data,
                                 std::size_t nbytes);

  [[nodiscard]] Error set_tensor_f32(const std::string& name_or_id,
                                     const std::vector<float>& values);

  [[nodiscard]] Error get_tensor(const std::string& name_or_id,
                                 void* data,
                                 std::size_t nbytes) const;

  [[nodiscard]] Error get_tensor_f32(const std::string& name_or_id,
                                     std::vector<float>* values) const;

  [[nodiscard]] const memory::TensorBuffer* find_buffer(TensorId id) const;
  [[nodiscard]] const memory::TensorBuffer* find_buffer(const std::string& name) const;

  [[nodiscard]] Error run();

  /// Called once per newly generated token (not prompt). Return false to stop early.
  using OnNewToken = std::function<bool(std::int64_t token_id)>;

  /// Autoregressive generate: prefill prompt, then decode up to max_new_tokens.
  /// Sampling: temperature / top-k / top-p / repetition_penalty (see SampleParams).
  /// Prefer graph `logits` when sampling; `probs` ok for greedy.
  /// Stops early when a token is in stop_token_ids (token is included in out_tokens).
  [[nodiscard]] Error generate(const std::vector<std::int64_t>& prompt_tokens,
                               std::int64_t max_new_tokens,
                               std::vector<std::int64_t>* out_tokens,
                               const std::vector<std::int64_t>& stop_token_ids = {},
                               const OnNewToken& on_new_token = {},
                               const SampleParams& sample = {});

  [[nodiscard]] KvCache* kv_cache() noexcept { return kv_.get(); }
  [[nodiscard]] const KvCache* kv_cache() const noexcept { return kv_.get(); }

  [[nodiscard]] std::string debug_stats() const;

 private:
  [[nodiscard]] Error allocate_all_tensors();
  [[nodiscard]] Error load_or_init_weights();
  [[nodiscard]] Error resolve_tensor(const std::string& name_or_id, TensorId* out) const;
  [[nodiscard]] kernels::TensorView view_of(memory::TensorBuffer& buf) const;
  [[nodiscard]] Error stage_streamed_inputs(const ir::Node& node);
  void try_prefetch_stream_h2d(TensorId tid);
  [[nodiscard]] Error ensure_stream_device(TensorId tid, std::size_t nbytes);
  [[nodiscard]] Error upload_streamed_weight(TensorId tid, const void* host,
                                             std::size_t nbytes);
  [[nodiscard]] Error sync_stream_h2d(TensorId tid);
  [[nodiscard]] Error ensure_kv_cache(std::int64_t max_seq);
  [[nodiscard]] Error write_tokens_step(TensorId tokens_id, std::int64_t token);
  [[nodiscard]] Error read_argmax_token(TensorId scores_id, std::int64_t* token) const;
  [[nodiscard]] Error read_sample_token(TensorId scores_id,
                                        const SampleParams& sample,
                                        const std::vector<std::int64_t>& history,
                                        std::mt19937_64* rng,
                                        std::int64_t* token) const;

  bool ready_ = false;
  SessionOptions options_;
  ir::Graph graph_;
  ir::ExecutionPlan plan_;
  OptimizeReport report_;
  std::unique_ptr<memory::Allocator> allocator_;
  std::unique_ptr<IBackend> backend_;
  DeviceScheduler scheduler_;
  profiler::Profiler profiler_;
  std::unique_ptr<storage::StreamingWeightStore> streaming_;
  struct StreamDeviceStaging {
    void* device_ptr = nullptr;
    std::size_t nbytes = 0;
    bool h2d_in_flight = false;
  };
  std::unordered_map<TensorId, StreamDeviceStaging> stream_device_;
  bool stream_async_h2d_ = false;
  std::unique_ptr<KvCache> kv_;
  planner::MemoryReusePlan memory_plan_;
  std::unordered_map<int, memory::TensorBuffer> owned_slots_;
  std::unordered_map<TensorId, memory::TensorBuffer> buffers_;
  std::unordered_map<std::string, TensorId> name_to_id_;
};

[[nodiscard]] UAII_API Error run_toy_mlp_demo(std::vector<float>* out_values,
                                              bool* matched_expected);

[[nodiscard]] UAII_API Error run_tiny_block_demo(std::vector<float>* out_values);

/// Write the Phase-4 tiny LM GGUF used by demos; returns path on success.
[[nodiscard]] UAII_API Error materialize_tiny_gguf_demo(std::string* out_path);

/// Phase 4: write tiny GGUF → load → generate-style forward; checks softmax mass.
[[nodiscard]] UAII_API Error run_gguf_generate_demo(std::string* decoded,
                                                    bool* ok);

/// Phase 4: write tiny Safetensors → load → generate-style forward.
[[nodiscard]] UAII_API Error run_safetensors_generate_demo(std::string* decoded,
                                                           bool* ok);

/// Phase 4: MoE router + expert dispatch smoke test.
[[nodiscard]] UAII_API Error run_moe_smoke_demo(bool* ok);

/// Phase 5: run the same IR graph on two backends and compare under parity policy.
[[nodiscard]] UAII_API Error run_backend_parity(const ir::Graph& graph,
                                                const std::string& backend_a,
                                                const std::string& backend_b,
                                                const backends::ParityPolicy& policy,
                                                backends::ParityReport* report,
                                                const std::vector<float>& input_f32 = {
                                                    1.f, 2.f, 3.f, 4.f},
                                                const std::string& input_name = "x");

/// Phase 5 demo: toy MLP on cpu vs cuda with parity policy (native CUDA when available).
[[nodiscard]] UAII_API Error run_parity_demo(backends::ParityReport* report);

struct OptimizeDemoReport {
  OptimizeReport baseline;
  OptimizeReport optimized;
  bool ok = false;
  std::string message;
};

/// Phase 6: fusion + memory reuse wins vs baseline.
[[nodiscard]] UAII_API Error run_optimize_demo(OptimizeDemoReport* report);

struct StreamingDemoReport {
  std::uint64_t total_weight_bytes = 0;
  std::uint64_t ram_budget_bytes = 0;
  std::uint64_t staging_bytes = 0;
  std::uint64_t bytes_read = 0;
  bool ok = false;
  std::string message;
};

/// Phase 6: streaming path for weights larger than RAM budget (controlled fixture).
[[nodiscard]] UAII_API Error run_streaming_demo(StreamingDemoReport* report);

struct ProfileDemoReport {
  std::string trace_path;
  std::string summary;
  bool ok = false;
};

/// Phase 6: run with profiler and write chrome-trace JSON.
[[nodiscard]] UAII_API Error run_profile_demo(const std::string& trace_path,
                                              ProfileDemoReport* report);

struct QuantDemoReport {
  quant::QuantFormat format = quant::QuantFormat::INT8;
  float max_abs_err = 0;
  bool ok = false;
  std::string message;
};

/// Phase 6: pack/unpack roundtrip for supported quant formats.
[[nodiscard]] UAII_API Error run_quant_demo(quant::QuantFormat format,
                                            QuantDemoReport* report);

}  // namespace runtime
}  // namespace uaii
