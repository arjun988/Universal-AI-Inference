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
#include "uaii/runtime/scheduler_cpu.hpp"
#include "uaii/storage/streaming.hpp"

#include <memory>
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

  [[nodiscard]] std::string debug_stats() const;

 private:
  [[nodiscard]] Error allocate_all_tensors();
  [[nodiscard]] Error load_or_init_weights();
  [[nodiscard]] Error resolve_tensor(const std::string& name_or_id, TensorId* out) const;
  [[nodiscard]] kernels::TensorView view_of(memory::TensorBuffer& buf) const;
  [[nodiscard]] Error stage_streamed_inputs(const ir::Node& node);

  bool ready_ = false;
  SessionOptions options_;
  ir::Graph graph_;
  ir::ExecutionPlan plan_;
  OptimizeReport report_;
  std::unique_ptr<memory::Allocator> allocator_;
  std::unique_ptr<IBackend> backend_;
  CpuScheduler scheduler_;
  profiler::Profiler profiler_;
  std::unique_ptr<storage::StreamingWeightStore> streaming_;
  planner::MemoryReusePlan memory_plan_;
  std::unordered_map<int, memory::TensorBuffer> owned_slots_;
  std::unordered_map<TensorId, memory::TensorBuffer> buffers_;
  std::unordered_map<std::string, TensorId> name_to_id_;
};

[[nodiscard]] UAII_API Error run_toy_mlp_demo(std::vector<float>* out_values,
                                              bool* matched_expected);

[[nodiscard]] UAII_API Error run_tiny_block_demo(std::vector<float>* out_values);

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

/// Phase 5 demo: toy MLP on cpu vs cuda (host-fallback) with parity policy.
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
