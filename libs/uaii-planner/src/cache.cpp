#include "uaii/planner/cache.hpp"

#include <functional>
#include <sstream>

namespace uaii {
namespace planner {

PlanCache& PlanCache::instance() {
  static PlanCache cache;
  return cache;
}

bool PlanCache::get(const std::string& key, ir::ExecutionPlan* out) const {
  if (out == nullptr) return false;
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return false;
  *out = it->second;
  return true;
}

void PlanCache::put(std::string key, ir::ExecutionPlan plan) {
  std::lock_guard<std::mutex> lock(mu_);
  entries_[std::move(key)] = std::move(plan);
}

void PlanCache::clear() noexcept {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.clear();
}

std::size_t PlanCache::size() const noexcept {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.size();
}

std::string PlanCache::fingerprint(const ir::Graph& graph, bool fusion_enabled) {
  std::ostringstream oss;
  oss << graph.name << "|" << graph.nodes.size() << "|" << graph.tensors.size()
      << "|f=" << (fusion_enabled ? 1 : 0);
  for (const auto& n : graph.nodes) {
    oss << ";" << n.id << ":" << n.op_name << "@" << n.op_version;
    for (TensorId id : n.inputs) oss << "<" << id;
    for (TensorId id : n.outputs) oss << ">" << id;
  }
  for (const auto& t : graph.tensors) {
    oss << "|t" << t.id << ":" << static_cast<int>(t.dtype);
    for (auto d : t.shape.dims) oss << "x" << d;
  }
  // Stable hash string
  const std::string raw = oss.str();
  const std::size_t h = std::hash<std::string>{}(raw);
  return std::to_string(h) + ":" + std::to_string(raw.size());
}

}  // namespace planner
}  // namespace uaii
