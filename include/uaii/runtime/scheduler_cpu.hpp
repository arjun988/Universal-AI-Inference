#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/scheduler.hpp"
#include "uaii/ir/plan.hpp"

namespace uaii {
namespace runtime {

/// Places every node on CPU; ordering comes from the execution plan.
class UAII_API CpuScheduler : public IScheduler {
 public:
  [[nodiscard]] std::string name() const override { return "cpu"; }

  [[nodiscard]] Error schedule(const std::vector<NodeId>& nodes,
                               std::vector<ScheduleDecision>* out) override;

  /// Fill decisions for a full execution plan (topo order preserved).
  [[nodiscard]] Error schedule_plan(const ir::ExecutionPlan& plan,
                                    std::vector<ScheduleDecision>* out);
};

}  // namespace runtime
}  // namespace uaii
