#pragma once

#include "uaii/backends/cpu_backend.hpp"
#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/ir/plan.hpp"
#include "uaii/kernels/tensor_view.hpp"
#include "uaii/memory/allocator.hpp"
#include "uaii/memory/buffer.hpp"
#include "uaii/runtime/scheduler_cpu.hpp"

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
  bool validate = true;
  bool allow_unknown_ops = false;
};

/// End-to-end CPU inference session over a UAII IR graph.
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

  bool ready_ = false;
  SessionOptions options_;
  ir::Graph graph_;
  ir::ExecutionPlan plan_;
  std::unique_ptr<memory::Allocator> allocator_;
  std::unique_ptr<backends::CpuBackend> backend_;
  CpuScheduler scheduler_;
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

}  // namespace runtime
}  // namespace uaii
