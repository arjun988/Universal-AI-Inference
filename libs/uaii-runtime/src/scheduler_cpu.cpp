#include "uaii/runtime/scheduler_cpu.hpp"

namespace uaii {
namespace runtime {

Error CpuScheduler::schedule(const std::vector<NodeId>& nodes,
                             std::vector<ScheduleDecision>* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "schedule out is null");
  }
  out->clear();
  out->reserve(nodes.size());
  int priority = static_cast<int>(nodes.size());
  for (NodeId id : nodes) {
    ScheduleDecision d;
    d.node_id = id;
    d.device = DeviceType::Cpu;
    d.priority = priority--;
    d.reason = "cpu-default";
    out->push_back(std::move(d));
  }
  return Error::ok();
}

Error CpuScheduler::schedule_plan(const ir::ExecutionPlan& plan,
                                  std::vector<ScheduleDecision>* out) {
  std::vector<NodeId> nodes;
  nodes.reserve(plan.ops.size());
  for (const auto& op : plan.ops) {
    nodes.push_back(op.node_id);
  }
  return schedule(nodes, out);
}

}  // namespace runtime
}  // namespace uaii
