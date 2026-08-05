#include "uaii/runtime/scheduler_device.hpp"

namespace uaii {
namespace runtime {

Error DeviceScheduler::schedule_plan(const ir::ExecutionPlan& plan,
                                     std::vector<ScheduleDecision>* out) const {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "schedule out null");
  }
  out->clear();
  out->reserve(plan.ops.size());
  int priority = static_cast<int>(plan.ops.size());
  for (const auto& op : plan.ops) {
    ScheduleDecision d;
    d.node_id = op.node_id;
    d.priority = priority--;
    const bool attn = op.op_name == "Attention" || op.op_name == "RoPE";
    if (preferred_ == DeviceType::Cpu) {
      d.device = DeviceType::Cpu;
      d.reason = "cpu";
    } else if (attn && attention_host_fallback_) {
      d.device = DeviceType::Cpu;
      d.reason = "attention-host-fallback";
    } else {
      d.device = preferred_;
      d.reason = "device-preferred";
    }
    out->push_back(std::move(d));
  }
  return Error::success();
}

}  // namespace runtime
}  // namespace uaii
