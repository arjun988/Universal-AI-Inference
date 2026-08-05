#include "uaii/planner/cache.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

namespace uaii {
namespace planner {
namespace {

std::string cache_dir() {
  const char* env = std::getenv("UAII_PLAN_CACHE_DIR");
  if (env && env[0] != '\0') return env;
  return {};
}

void write_id_list(std::ostream& out, const std::vector<std::uint64_t>& ids) {
  out << ids.size();
  for (auto id : ids) out << ' ' << id;
  out << '\n';
}

bool read_id_list(std::istream& in, std::vector<std::uint64_t>* ids) {
  std::string line;
  if (!std::getline(in, line)) return false;
  std::istringstream ss(line);
  std::size_t n = 0;
  if (!(ss >> n)) return false;
  ids->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!(ss >> (*ids)[i])) return false;
  }
  return true;
}

bool write_plan(const std::string& path, const ir::ExecutionPlan& plan) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "UAIIPLAN1\n";
  out << plan.graph_name << '\n';
  out << plan.ir_version.major << ' ' << plan.ir_version.minor << '\n';
  out << plan.ops.size() << '\n';
  for (const auto& op : plan.ops) {
    out << op.node_id << '\n';
    out << op.node_name << '\n';
    out << op.op_name << '\n';
    out << op.op_version << '\n';
    out << static_cast<int>(op.preferred_device) << '\n';
    out << op.selected_kernel << '\n';
    write_id_list(out, op.dependencies);
    write_id_list(out, op.inputs);
    write_id_list(out, op.outputs);
  }
  out << plan.memory_hints.size() << '\n';
  for (const auto& h : plan.memory_hints) {
    out << h.tensor_id << '\n';
    out << h.name << '\n';
    out << h.estimated_bytes << '\n';
    out << (h.is_graph_input ? 1 : 0) << ' ' << (h.is_graph_output ? 1 : 0) << ' '
        << (h.is_weight ? 1 : 0) << '\n';
  }
  write_id_list(out, plan.graph_inputs);
  write_id_list(out, plan.graph_outputs);
  return static_cast<bool>(out);
}

bool read_plan(const std::string& path, ir::ExecutionPlan* out_plan) {
  std::ifstream in(path, std::ios::binary);
  if (!in || out_plan == nullptr) return false;
  std::string magic;
  if (!std::getline(in, magic) || magic != "UAIIPLAN1") return false;
  ir::ExecutionPlan plan;
  if (!std::getline(in, plan.graph_name)) return false;
  if (!(in >> plan.ir_version.major >> plan.ir_version.minor)) return false;
  std::size_t nops = 0;
  if (!(in >> nops)) return false;
  in.get();  // newline
  plan.ops.reserve(nops);
  for (std::size_t i = 0; i < nops; ++i) {
    ir::PlannedOp op;
    int device = 0;
    if (!(in >> op.node_id)) return false;
    in.get();
    if (!std::getline(in, op.node_name)) return false;
    if (!std::getline(in, op.op_name)) return false;
    if (!std::getline(in, op.op_version)) return false;
    if (!(in >> device)) return false;
    in.get();
    if (!std::getline(in, op.selected_kernel)) return false;
    op.preferred_device = static_cast<DeviceType>(device);
    if (!read_id_list(in, &op.dependencies)) return false;
    if (!read_id_list(in, &op.inputs)) return false;
    if (!read_id_list(in, &op.outputs)) return false;
    plan.ops.push_back(std::move(op));
  }
  std::size_t nh = 0;
  if (!(in >> nh)) return false;
  in.get();
  plan.memory_hints.reserve(nh);
  for (std::size_t i = 0; i < nh; ++i) {
    ir::MemoryPlanHint h;
    int a = 0, b = 0, c = 0;
    if (!(in >> h.tensor_id)) return false;
    in.get();
    if (!std::getline(in, h.name)) return false;
    if (!(in >> h.estimated_bytes >> a >> b >> c)) return false;
    in.get();
    h.is_graph_input = a != 0;
    h.is_graph_output = b != 0;
    h.is_weight = c != 0;
    plan.memory_hints.push_back(std::move(h));
  }
  if (!read_id_list(in, &plan.graph_inputs)) return false;
  if (!read_id_list(in, &plan.graph_outputs)) return false;
  *out_plan = std::move(plan);
  return true;
}

}  // namespace

PlanCache& PlanCache::instance() {
  static PlanCache cache;
  return cache;
}

bool PlanCache::get(const std::string& key, ir::ExecutionPlan* out) const {
  if (out == nullptr) return false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      *out = it->second;
      return true;
    }
  }
  const std::string dir = cache_dir();
  if (dir.empty()) return false;
  ir::ExecutionPlan plan;
  if (!read_plan(dir + "/" + key + ".plan", &plan)) return false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    entries_[key] = plan;
  }
  *out = std::move(plan);
  return true;
}

void PlanCache::put(std::string key, ir::ExecutionPlan plan) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    entries_[key] = plan;
  }
  const std::string dir = cache_dir();
  if (dir.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  (void)write_plan(dir + "/" + key + ".plan", plan);
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
  oss << "v2|" << graph.name << "|" << graph.nodes.size() << "|" << graph.tensors.size()
      << "|f=" << (fusion_enabled ? 1 : 0) << "|ver=" << graph.version.major << "."
      << graph.version.minor;
  for (const auto& n : graph.nodes) {
    oss << ";n" << n.id << ":" << n.op_name << "@" << n.op_version;
    for (TensorId id : n.inputs) oss << "<" << id;
    for (TensorId id : n.outputs) oss << ">" << id;
    for (const auto& a : n.attributes) {
      oss << "|a" << a.key << "=" << static_cast<int>(a.type);
    }
  }
  for (const auto& t : graph.tensors) {
    oss << "|t" << t.id << ":" << static_cast<int>(t.dtype) << ":w" << (t.is_weight ? 1 : 0);
    for (auto d : t.shape.dims) oss << "x" << d;
    if (!t.weight_ref.empty()) oss << "#ref:" << t.weight_ref;
  }
  const std::string raw = oss.str();
  const std::size_t h1 = std::hash<std::string>{}(raw);
  const std::size_t h2 = std::hash<std::string>{}(raw + "#salt");
  std::ostringstream key;
  key << std::hex << h1 << h2 << "_" << raw.size();
  return key.str();
}

}  // namespace planner
}  // namespace uaii
