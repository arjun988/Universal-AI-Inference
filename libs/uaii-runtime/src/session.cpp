#include "uaii/runtime/session.hpp"

#include "uaii/backends/factory.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/dtype.hpp"
#include "uaii/ir/registry.hpp"
#include "uaii/ir/validator.hpp"
#include "uaii/kernels/kernels.hpp"
#include "uaii/loaders/registry.hpp"
#include "uaii/quant/quantizer.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace uaii {
namespace runtime {
namespace {

std::string join_path(const std::string& dir, const std::string& file) {
  if (dir.empty()) {
    return file;
  }
#if defined(_WIN32)
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  if (dir.back() == '/' || dir.back() == '\\') {
    return dir + file;
  }
  return dir + sep + file;
}

bool path_under_root(const std::string& root, const std::string& path) {
  if (root.empty()) return true;
  try {
    namespace fs = std::filesystem;
    const fs::path r = fs::weakly_canonical(fs::absolute(root));
    const fs::path p = fs::weakly_canonical(fs::absolute(path));
    auto rit = r.begin();
    auto pit = p.begin();
    for (; rit != r.end() && pit != p.end(); ++rit, ++pit) {
      if (*rit != *pit) return false;
    }
    return rit == r.end();
  } catch (...) {
    return false;
  }
}

}  // namespace

Session::Session() = default;

Session::~Session() {
  destroy();
}

Error Session::create(ir::Graph graph, SessionOptions options) {
  destroy();
  options_ = std::move(options);

  if (options_.enable_profiler) {
    profiler_.begin_session("uaii-session");
  }

  planner::OptimizeOptions oopts;
  oopts.enable_fusion = options_.enable_fusion;
  oopts.enable_memory_reuse = options_.enable_memory_reuse;
  oopts.enable_storage_plan = options_.enable_storage_plan;
  oopts.enable_plan_cache = options_.enable_plan_cache;
  oopts.enable_streaming = options_.enable_streaming;
  oopts.ram_budget_bytes = options_.allocator.budget_bytes;
  oopts.prefer_mmap = true;

  planner::OptimizeResult opt;
  {
    profiler::Profiler::Scope scope(
        options_.enable_profiler ? &profiler_ : nullptr, "optimize_graph",
        profiler::EventCategory::Planner);
    Error err = planner::optimize_graph(std::move(graph), oopts, &opt);
    if (!err.ok()) {
      destroy();
      return err;
    }
  }

  graph_ = std::move(opt.graph);
  plan_ = std::move(opt.plan);
  memory_plan_ = opt.memory;
  report_.summary = opt.summary;
  report_.fusion = opt.fusion;
  report_.memory = opt.memory;
  report_.storage = opt.storage;
  report_.cache_hit = opt.cache_hit;

  if (options_.validate) {
    ir::ValidationOptions vopts;
    vopts.allow_unknown_ops = options_.allow_unknown_ops;
    Error err = ir::validate_graph_error(graph_, ir::default_registry(), vopts);
    if (!err.ok()) {
      destroy();
      return err;
    }
  }

  for (const auto& n : graph_.nodes) {
    if (!kernels::supports_cpu_op(n.op_name, n.op_version)) {
      destroy();
      return Error::make(ErrorCode::NotImplemented,
                         "no CPU kernel for " + n.op_name + "@" + n.op_version);
    }
  }

  allocator_ = std::make_unique<memory::Allocator>(options_.allocator);

  backends::BackendCreateOptions bopts;
  bopts.allocator = allocator_.get();
  bopts.prefer_native = options_.prefer_native;
  bopts.force_host_fallback = options_.force_host_fallback;
  Error err = backends::create_backend(options_.backend_name, bopts, &backend_);
  if (!err.ok()) {
    destroy();
    return err;
  }
  err = backend_->initialize();
  if (!err.ok()) {
    destroy();
    return err;
  }

  for (const auto& t : graph_.tensors) {
    if (!t.name.empty()) {
      name_to_id_[t.name] = t.id;
    }
  }

  if (options_.enable_streaming && report_.storage.streaming_required) {
    streaming_ = std::make_unique<storage::StreamingWeightStore>();
    err = streaming_->configure(report_.storage, options_.weights_dir);
    if (!err.ok()) {
      destroy();
      return err;
    }
  }

  {
    profiler::Profiler::Scope scope(
        options_.enable_profiler ? &profiler_ : nullptr, "allocate",
        profiler::EventCategory::Memory);
    err = allocate_all_tensors();
  }
  if (!err.ok()) {
    destroy();
    return err;
  }

  {
    profiler::Profiler::Scope scope(
        options_.enable_profiler ? &profiler_ : nullptr, "load_weights",
        profiler::EventCategory::Io);
    err = load_or_init_weights();
  }
  if (!err.ok()) {
    destroy();
    return err;
  }

  ready_ = true;
  log::info("session") << "created graph='" << graph_.name
                       << "' backend=" << backend_->name()
                       << " ops=" << plan_.ops.size() << " " << report_.summary
                       << " " << allocator_->stats();
  return Error::success();
}

void Session::destroy() noexcept {
  for (auto& kv : owned_slots_) {
    if (allocator_ && kv.second.owned && kv.second.data != nullptr) {
      allocator_->deallocate_bytes(kv.second.data, kv.second.nbytes);
      kv.second.data = nullptr;
    }
  }
  owned_slots_.clear();
  for (auto& kv : buffers_) {
    if (allocator_ && kv.second.owned && kv.second.data != nullptr) {
      allocator_->deallocate_tensor(&kv.second);
    }
  }
  buffers_.clear();
  name_to_id_.clear();
  streaming_.reset();
  if (backend_) {
    backend_->shutdown();
    backend_.reset();
  }
  if (allocator_) {
    allocator_->reset();
    allocator_.reset();
  }
  if (options_.enable_profiler && profiler_.active()) {
    profiler_.end_session();
  }
  plan_ = ir::ExecutionPlan{};
  graph_ = ir::Graph{};
  memory_plan_ = planner::MemoryReusePlan{};
  report_ = OptimizeReport{};
  ready_ = false;
}

Error Session::allocate_all_tensors() {
  const bool reuse = options_.enable_memory_reuse && !memory_plan_.slots.empty();

  if (reuse) {
    for (const auto& slot : memory_plan_.slots) {
      // Skip allocating owned RAM for streamed weights — staging owns residency.
      bool streamed = streaming_ && !slot.tenants.empty() &&
                      streaming_->is_streamed(slot.tenants.front());
      if (streamed) {
        continue;
      }
      memory::TensorBuffer buf;
      buf.id = slot.tenants.empty() ? 0 : slot.tenants.front();
      buf.dtype = DType::F32;
      buf.nbytes = static_cast<std::size_t>(slot.bytes);
      Error err = allocator_->allocate_bytes(buf.nbytes, &buf.data);
      if (!err.ok()) return err;
      buf.owned = true;
      std::memset(buf.data, 0, buf.nbytes);
      owned_slots_.emplace(slot.slot_id, std::move(buf));
    }

    for (const auto& t : graph_.tensors) {
      // Compute path is f32; F16/BF16/INT* graph dtypes are coerced to f32 buffers
      // (weights dequantized on load). Reject unknown/unsupported types.
      if (t.dtype != DType::F32 && t.dtype != DType::F16 && t.dtype != DType::BF16 &&
          t.dtype != DType::I8 && t.dtype != DType::I32 && t.dtype != DType::U8) {
        return Error::make(ErrorCode::NotImplemented,
                           "session cannot coerce dtype for tensor " + t.name);
      }
      memory::TensorBuffer view;
      view.id = t.id;
      view.dtype = DType::F32;
      view.shape = t.shape;
      {
        const std::size_t n = ir::shape_numel(t.shape);
        view.nbytes = n > 0 ? n * sizeof(float)
                            : static_cast<std::size_t>(ir::estimate_tensor_bytes(t));
        if (n == 0 && t.dtype != DType::F32 && view.nbytes > 0) {
          const std::size_t elem = ir::dtype_size_bytes(t.dtype);
          if (elem > 0) view.nbytes = (view.nbytes / elem) * sizeof(float);
        }
      }
      view.owned = false;

      if (streaming_ && streaming_->is_streamed(t.id)) {
        view.data = nullptr;  // filled on stage
        buffers_.emplace(t.id, std::move(view));
        continue;
      }

      auto sit = memory_plan_.tensor_to_slot.find(t.id);
      if (sit == memory_plan_.tensor_to_slot.end()) {
        return Error::make(ErrorCode::Internal, "missing slot for tensor");
      }
      auto bit = owned_slots_.find(sit->second);
      if (bit == owned_slots_.end()) {
        return Error::make(ErrorCode::Internal, "missing owned slot");
      }
      view.data = bit->second.data;
      buffers_.emplace(t.id, std::move(view));
    }
    return Error::success();
  }

  // Naive path: one buffer per tensor (skip streamed weights); always f32 storage.
  for (const auto& t : graph_.tensors) {
    if (t.dtype != DType::F32 && t.dtype != DType::F16 && t.dtype != DType::BF16 &&
        t.dtype != DType::I8 && t.dtype != DType::I32 && t.dtype != DType::U8) {
      return Error::make(ErrorCode::NotImplemented,
                         "session cannot coerce dtype for tensor " + t.name);
    }
    memory::TensorBuffer buf;
    if (streaming_ && streaming_->is_streamed(t.id)) {
      buf.id = t.id;
      buf.dtype = DType::F32;
      buf.shape = t.shape;
      buf.nbytes = ir::shape_numel(t.shape) * sizeof(float);
      buf.data = nullptr;
      buf.owned = false;
      buffers_.emplace(t.id, std::move(buf));
      continue;
    }
    ir::Tensor coerced = t;
    coerced.dtype = DType::F32;
    Error err = allocator_->allocate_tensor(coerced, &buf);
    if (!err.ok()) return err;
    std::memset(buf.data, 0, buf.nbytes);
    buffers_.emplace(t.id, std::move(buf));
  }
  return Error::success();
}

Error Session::load_or_init_weights() {
  for (const auto& t : graph_.tensors) {
    if (!t.is_weight) continue;
    if (streaming_ && streaming_->is_streamed(t.id)) {
      continue;  // staged on demand
    }
    auto it = buffers_.find(t.id);
    if (it == buffers_.end()) {
      return Error::make(ErrorCode::Internal, "missing weight buffer");
    }
    memory::TensorBuffer& buf = it->second;
    float* data = memory::as_f32(buf);
    const std::size_t n = buf.nbytes / sizeof(float);

    bool loaded = false;
    Error load_err = Error::success();
    if (!t.weight_ref.empty()) {
      load_err = loaders::load_weight_ref_f32(t.weight_ref, options_.weights_dir,
                                              t.shape, data, buf.nbytes);
      if (load_err.ok()) {
        loaded = true;
      } else {
        std::string path = join_path(options_.weights_dir, t.weight_ref);
        const auto hash = path.find('#');
        if (hash != std::string::npos) path = path.substr(0, hash);
        if (!path_under_root(options_.weights_sandbox.empty() ? options_.weights_dir
                                                              : options_.weights_sandbox,
                             path)) {
          return Error::make(ErrorCode::InvalidArgument,
                             "weight path escapes sandbox: " + path);
        }
        std::ifstream in(path, std::ios::binary);
        if (in) {
          if (options_.weight_quant != quant::QuantFormat::F32) {
            std::vector<std::uint8_t> packed(
                (std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
            quant::QuantParams qp;
            qp.group_size = 32;
            auto* q = quant::QuantizerRegistry::instance().default_quantizer();
            if (q == nullptr) {
              return Error::make(ErrorCode::Internal, "no quantizer");
            }
            Error uerr =
                q->unpack(packed.data(), packed.size(), n, options_.weight_quant, qp,
                          nullptr, 0, data);
            if (uerr.ok()) loaded = true;
            else load_err = uerr;
          } else {
            in.read(reinterpret_cast<char*>(data),
                    static_cast<std::streamsize>(buf.nbytes));
            if (static_cast<std::size_t>(in.gcount()) == buf.nbytes) {
              loaded = true;
            } else {
              load_err = Error::make(ErrorCode::IoError, "short weight read: " + path);
            }
          }
        }
      }
    }

    if (loaded) continue;

    // Fail closed: missing weight_ref with weight_init=None is an error.
    // Explicit weight_init (zeros/ones/sequence) is an opt-in synthetic fill.
    if (!t.weight_ref.empty() && options_.weight_init == WeightInit::None &&
        !options_.allow_missing_weights) {
      return Error::make(ErrorCode::NotFound,
                         "failed to load required weight '" + t.name + "' from '" +
                             t.weight_ref + "': " + load_err.to_string());
    }

    switch (options_.weight_init) {
      case WeightInit::Zeros:
        std::memset(data, 0, buf.nbytes);
        break;
      case WeightInit::Ones:
        for (std::size_t i = 0; i < n; ++i) data[i] = 1.0f;
        break;
      case WeightInit::Sequence:
        for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<float>(i) * 0.01f;
        break;
      case WeightInit::None:
        if (t.weight_ref.empty()) {
          return Error::make(ErrorCode::NotFound,
                             "weight tensor '" + t.name +
                                 "' has no weight_ref and weight_init=None");
        }
        return Error::make(ErrorCode::NotFound,
                           "failed to load weight '" + t.name + "' from '" +
                               t.weight_ref + "'");
    }
  }
  return Error::success();
}

Error Session::resolve_tensor(const std::string& name_or_id, TensorId* out) const {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tensor id out null");
  }
  if (!name_or_id.empty() && name_or_id[0] == '#') {
    try {
      *out = static_cast<TensorId>(std::stoull(name_or_id.substr(1)));
      return Error::success();
    } catch (...) {
      return Error::make(ErrorCode::InvalidArgument, "bad tensor id " + name_or_id);
    }
  }
  auto it = name_to_id_.find(name_or_id);
  if (it != name_to_id_.end()) {
    *out = it->second;
    return Error::success();
  }
  try {
    *out = static_cast<TensorId>(std::stoull(name_or_id));
    if (buffers_.count(*out) != 0) {
      return Error::success();
    }
  } catch (...) {
  }
  return Error::make(ErrorCode::NotFound, "tensor not found: " + name_or_id);
}

const memory::TensorBuffer* Session::find_buffer(TensorId id) const {
  auto it = buffers_.find(id);
  if (it == buffers_.end()) return nullptr;
  return &it->second;
}

const memory::TensorBuffer* Session::find_buffer(const std::string& name) const {
  TensorId id = 0;
  if (!resolve_tensor(name, &id).ok()) return nullptr;
  return find_buffer(id);
}

kernels::TensorView Session::view_of(memory::TensorBuffer& buf) const {
  kernels::TensorView v;
  v.dtype = buf.dtype;
  v.shape = buf.shape.dims.data();
  v.rank = buf.shape.dims.size();
  v.data = buf.data;
  v.nbytes = buf.nbytes;
  return v;
}

Error Session::stage_streamed_inputs(const ir::Node& node) {
  if (!streaming_) return Error::success();
  for (TensorId tid : node.inputs) {
    if (!streaming_->is_streamed(tid)) continue;
    const void* data = nullptr;
    std::size_t nbytes = 0;
    profiler::Profiler::Scope scope(
        options_.enable_profiler ? &profiler_ : nullptr, "stream_weight",
        profiler::EventCategory::Io, "tensor=" + std::to_string(tid));
    Error err = streaming_->stage(tid, &data, &nbytes);
    if (!err.ok()) return err;
    auto it = buffers_.find(tid);
    if (it == buffers_.end()) {
      return Error::make(ErrorCode::Internal, "missing streamed buffer");
    }
    it->second.data = const_cast<void*>(data);
    it->second.nbytes = nbytes;
  }
  return Error::success();
}

Error Session::set_tensor(const std::string& name_or_id,
                          const void* data,
                          std::size_t nbytes) {
  if (!ready_) {
    return Error::make(ErrorCode::InvalidArgument, "session not ready");
  }
  TensorId id = 0;
  Error err = resolve_tensor(name_or_id, &id);
  if (!err.ok()) return err;
  auto it = buffers_.find(id);
  if (it == buffers_.end()) {
    return Error::make(ErrorCode::NotFound, "buffer missing");
  }
  auto& buf = it->second;
  if (buf.data == nullptr || data == nullptr || nbytes != buf.nbytes) {
    return Error::make(ErrorCode::InvalidArgument,
                       "set_tensor size mismatch for " + name_or_id);
  }
  std::memcpy(buf.data, data, nbytes);
  return Error::success();
}

Error Session::set_tensor_f32(const std::string& name_or_id,
                              const std::vector<float>& values) {
  return set_tensor(name_or_id, values.data(), values.size() * sizeof(float));
}

Error Session::get_tensor(const std::string& name_or_id,
                          void* data,
                          std::size_t nbytes) const {
  if (!ready_) {
    return Error::make(ErrorCode::InvalidArgument, "session not ready");
  }
  TensorId id = 0;
  Error err = resolve_tensor(name_or_id, &id);
  if (!err.ok()) return err;
  const auto* buf = find_buffer(id);
  if (buf == nullptr || buf->data == nullptr) {
    return Error::make(ErrorCode::NotFound, "buffer missing");
  }
  if (data == nullptr || nbytes != buf->nbytes) {
    return Error::make(ErrorCode::InvalidArgument, "get_tensor size mismatch");
  }
  std::memcpy(data, buf->data, nbytes);
  return Error::success();
}

Error Session::get_tensor_f32(const std::string& name_or_id,
                              std::vector<float>* values) const {
  if (values == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "values out null");
  }
  TensorId id = 0;
  Error err = resolve_tensor(name_or_id, &id);
  if (!err.ok()) return err;
  const auto* buf = find_buffer(id);
  if (buf == nullptr || buf->data == nullptr) {
    return Error::make(ErrorCode::NotFound, "buffer missing");
  }
  const std::size_t n = buf->nbytes / sizeof(float);
  values->resize(n);
  std::memcpy(values->data(), buf->data, buf->nbytes);
  return Error::success();
}

Error Session::run() {
  if (!ready_) {
    return Error::make(ErrorCode::InvalidArgument, "session not ready");
  }

  std::vector<ScheduleDecision> decisions;
  Error err = scheduler_.schedule_plan(plan_, &decisions);
  if (!err.ok()) return err;

  for (std::size_t oi = 0; oi < plan_.ops.size(); ++oi) {
    const auto& planned = plan_.ops[oi];
    const ir::Node* node = graph_.find_node(planned.node_id);
    if (node == nullptr) {
      return Error::make(ErrorCode::Internal,
                         "missing node " + std::to_string(planned.node_id));
    }

    // Apply scheduler placement: reject ops scheduled away from the active backend
    // unless the backend reports host-fallback (CPU kernels).
    if (oi < decisions.size() && backend_) {
      const auto& d = decisions[oi];
      if (d.device != DeviceType::Cpu && d.device != backend_->device_type() &&
          !backend_->uses_host_fallback()) {
        return Error::make(ErrorCode::NotImplemented,
                           "scheduler selected device " +
                               std::string(to_string(d.device)) +
                               " unavailable on backend " + backend_->name());
      }
    }

    err = stage_streamed_inputs(*node);
    if (!err.ok()) return err;

    // Prefetch streamed weights for the next op into the inactive staging buffer.
    if (streaming_ && oi + 1 < plan_.ops.size()) {
      const ir::Node* next = graph_.find_node(plan_.ops[oi + 1].node_id);
      if (next) {
        for (TensorId tid : next->inputs) {
          if (streaming_->is_streamed(tid)) streaming_->prefetch(tid);
        }
      }
    }

    std::vector<kernels::TensorView> inputs;
    inputs.reserve(node->inputs.size());
    for (TensorId tid : node->inputs) {
      auto it = buffers_.find(tid);
      if (it == buffers_.end() || it->second.data == nullptr) {
        return Error::make(ErrorCode::NotFound,
                           "missing input tensor " + std::to_string(tid));
      }
      inputs.push_back(view_of(it->second));
    }

    std::vector<kernels::TensorView> outputs;
    outputs.reserve(node->outputs.size());
    for (TensorId tid : node->outputs) {
      auto it = buffers_.find(tid);
      if (it == buffers_.end() || it->second.data == nullptr) {
        return Error::make(ErrorCode::NotFound,
                           "missing output tensor " + std::to_string(tid));
      }
      outputs.push_back(view_of(it->second));
    }

    {
      profiler::Profiler::Scope scope(
          options_.enable_profiler ? &profiler_ : nullptr, node->op_name,
          profiler::EventCategory::Kernel, "node=" + std::to_string(node->id));
      err = backend_->dispatch(node->op_name, node->op_version, inputs, &outputs,
                               node->attributes);
    }
    if (!err.ok()) {
      return Error::make(err.code(),
                         "op '" + node->op_name + "' node=" +
                             std::to_string(node->id) + ": " + err.message());
    }
  }

  err = backend_->synchronize();
  if (!err.ok()) return err;

  if (options_.enable_profiler && !options_.profile_trace_path.empty()) {
    Error terr = profiler::write_chrome_trace(profiler_, options_.profile_trace_path);
    if (!terr.ok()) return terr;
  }
  return Error::success();
}

std::string Session::debug_stats() const {
  std::ostringstream oss;
  oss << "graph=" << graph_.name << " ops=" << plan_.ops.size()
      << " tensors=" << buffers_.size();
  if (backend_) {
    oss << " backend=" << backend_->name();
    if (backend_->uses_host_fallback()) oss << "(host-fallback)";
  }
  if (!report_.summary.empty()) oss << " opt={" << report_.summary << "}";
  if (allocator_) oss << " " << allocator_->stats();
  if (options_.enable_profiler) oss << " profile={" << profiler_.summary() << "}";
  if (streaming_) {
    oss << " stream_io=" << streaming_->bytes_read() << "B";
  }
  return oss.str();
}

}  // namespace runtime
}  // namespace uaii
