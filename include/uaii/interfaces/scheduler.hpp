#pragma once

#include "uaii/core/error.hpp"
#include "uaii/interfaces/types.hpp"

#include <string>
#include <vector>

namespace uaii {

struct ScheduleDecision {
  NodeId node_id = 0;
  DeviceType device = DeviceType::Cpu;
  int priority = 0;
  std::string reason;
};

/// Decides where / when / how operators execute.
class IScheduler {
 public:
  virtual ~IScheduler() = default;

  [[nodiscard]] virtual std::string name() const = 0;

  /// Produce placement decisions for a set of node ids (Phase 3 expands).
  [[nodiscard]] virtual Error schedule(const std::vector<NodeId>& nodes,
                                       std::vector<ScheduleDecision>* out) = 0;
};

}  // namespace uaii
