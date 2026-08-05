#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/ir/plan.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace uaii {
namespace planner {

/// In-process plan cache keyed by graph fingerprint.
class UAII_API PlanCache {
 public:
  static PlanCache& instance();

  [[nodiscard]] bool get(const std::string& key, ir::ExecutionPlan* out) const;
  void put(std::string key, ir::ExecutionPlan plan);
  void clear() noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

  [[nodiscard]] static std::string fingerprint(const ir::Graph& graph,
                                               bool fusion_enabled);

 private:
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, ir::ExecutionPlan> entries_;
};

}  // namespace planner
}  // namespace uaii
