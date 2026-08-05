#pragma once

#include "uaii/export.hpp"
#include "uaii/runtime/scheduler_cpu.hpp"

#include <string>

namespace uaii {
namespace runtime {

/// Device-aware scheduler: places ops on `preferred` when backend can run them,
/// otherwise CPU. Used when Session backend is CUDA/Metal/etc.
class UAII_API DeviceScheduler {
 public:
  explicit DeviceScheduler(DeviceType preferred = DeviceType::Cpu)
      : preferred_(preferred) {}

  void set_preferred(DeviceType d) { preferred_ = d; }
  void set_attention_host_fallback(bool v) { attention_host_fallback_ = v; }

  [[nodiscard]] Error schedule_plan(const ir::ExecutionPlan& plan,
                                    std::vector<ScheduleDecision>* out) const;

 private:
  DeviceType preferred_ = DeviceType::Cpu;
  bool attention_host_fallback_ = false;
};

}  // namespace runtime
}  // namespace uaii
