#include "uaii/runtime/session.hpp"

#include "uaii/backends/factory.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/registry.hpp"
#include "uaii/ir/validator.hpp"
#include "uaii/kernels/kernels.hpp"
#include "uaii/loaders/registry.hpp"

#include <cstring>
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

}  // namespace

Session::Session() = default;

Session::~Session() {
  destroy();
}

Error Session::create(ir::Graph graph, SessionOptions options) {
  destroy();
  options_ = std::move(options);
  graph_ = std::move(graph);

  if (options_.validate) {
    ir::ValidationOptions vopts;
    vopts.allow_unknown_ops = options_.allow_unknown_ops;
    Error err = ir::validate_graph_error(graph_, ir::default_registry(), vopts);
    if (!err.ok()) {
      return err;
    }
  }

  for (const auto& n : graph_.nodes) {
    if (!kernels::supports_cpu_op(n.op_name, n.op_version)) {
      return Error::make(ErrorCode::NotImplemented,
                         "no CPU kernel for " + n.op_name + "@" + n.op_version);
    }
  }

  Error err = ir::build_execution_plan(graph_, &plan_);
  if (!err.ok()) {
    return err;
  }

  allocator_ = std::make_unique<memory::Allocator>(options_.allocator);

  backends::BackendCreateOptions bopts;
  bopts.allocator = allocator_.get();
  bopts.prefer_native = options_.prefer_native;
  bopts.force_host_fallback = options_.force_host_fallback;
  err = backends::create_backend(options_.backend_name, bopts, &backend_);
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

  err = allocate_all_tensors();
  if (!err.ok()) {
    destroy();
    return err;
  }

  err = load_or_init_weights();
  if (!err.ok()) {
    destroy();
    return err;
  }

  ready_ = true;
  log::info("session") << "created graph='" << graph_.name
                       << "' backend=" << backend_->name()
                       << " ops=" << plan_.ops.size() << " "
                       << allocator_->stats();
  return Error::ok();
}

void Session::destroy() noexcept {
  for (auto& kv : buffers_) {
    if (allocator_) {
      allocator_->deallocate_tensor(&kv.second);
    }
  }
  buffers_.clear();
  name_to_id_.clear();
  if (backend_) {
    backend_->shutdown();
    backend_.reset();
  }
  if (allocator_) {
    allocator_->reset();
    allocator_.reset();
  }
  plan_ = ir::ExecutionPlan{};
  graph_ = ir::Graph{};
  ready_ = false;
}

Error Session::allocate_all_tensors() {
  for (const auto& t : graph_.tensors) {
    if (t.dtype != DType::F32) {
      return Error::make(ErrorCode::NotImplemented,
                         "Phase 3 CPU runtime supports f32 only (tensor " + t.name +
                             ")");
    }
    memory::TensorBuffer buf;
    Error err = allocator_->allocate_tensor(t, &buf);
    if (!err.ok()) {
      return err;
    }
    std::memset(buf.data, 0, buf.nbytes);
    buffers_.emplace(t.id, std::move(buf));
  }
  return Error::ok();
}

Error Session::load_or_init_weights() {
  for (const auto& t : graph_.tensors) {
    if (!t.is_weight) {
      continue;
    }
    auto it = buffers_.find(t.id);
    if (it == buffers_.end()) {
      return Error::make(ErrorCode::Internal, "missing weight buffer");
    }
    memory::TensorBuffer& buf = it->second;
    float* data = memory::as_f32(buf);
    const std::size_t n = buf.nbytes / sizeof(float);

    bool loaded = false;
    if (!t.weight_ref.empty()) {
      Error werr = loaders::load_weight_ref_f32(t.weight_ref, options_.weights_dir,
                                               t.shape, data, buf.nbytes);
      if (werr.ok()) {
        loaded = true;
        log::debug("session") << "loaded weight " << t.name << " from "
                              << t.weight_ref;
      } else {
        // Fallback: raw binary path (legacy Phase 3)
        const std::string path = join_path(options_.weights_dir, t.weight_ref);
        std::ifstream in(path, std::ios::binary);
        if (in) {
          in.read(reinterpret_cast<char*>(data),
                  static_cast<std::streamsize>(buf.nbytes));
          if (static_cast<std::size_t>(in.gcount()) == buf.nbytes) {
            loaded = true;
            log::debug("session") << "loaded weight " << t.name << " from " << path;
          }
        }
        if (!loaded) {
          log::debug("session") << "weight load deferred/fail: " << werr.to_string();
        }
      }
    }

    if (loaded) {
      continue;
    }

    switch (options_.weight_init) {
      case WeightInit::Zeros:
        std::memset(data, 0, buf.nbytes);
        break;
      case WeightInit::Ones:
        for (std::size_t i = 0; i < n; ++i) {
          data[i] = 1.0f;
        }
        break;
      case WeightInit::Sequence:
        for (std::size_t i = 0; i < n; ++i) {
          data[i] = static_cast<float>(i) * 0.01f;
        }
        break;
      case WeightInit::None:
        if (t.weight_ref.empty()) {
          return Error::make(ErrorCode::NotFound,
                             "weight tensor '" + t.name +
                                 "' has no weight_ref and weight_init=None");
        }
        return Error::make(ErrorCode::NotFound,
                           "failed to load weight '" + t.name + "' from '" +
                               t.weight_ref +
                               "' (set --weight-init or --weights-dir)");
    }
  }
  return Error::ok();
}

Error Session::resolve_tensor(const std::string& name_or_id, TensorId* out) const {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tensor id out null");
  }
  if (!name_or_id.empty() && name_or_id[0] == '#') {
    try {
      *out = static_cast<TensorId>(std::stoull(name_or_id.substr(1)));
      return Error::ok();
    } catch (...) {
      return Error::make(ErrorCode::InvalidArgument, "bad tensor id " + name_or_id);
    }
  }
  auto it = name_to_id_.find(name_or_id);
  if (it != name_to_id_.end()) {
    *out = it->second;
    return Error::ok();
  }
  try {
    *out = static_cast<TensorId>(std::stoull(name_or_id));
    if (buffers_.count(*out) != 0) {
      return Error::ok();
    }
  } catch (...) {
  }
  return Error::make(ErrorCode::NotFound, "tensor not found: " + name_or_id);
}

const memory::TensorBuffer* Session::find_buffer(TensorId id) const {
  auto it = buffers_.find(id);
  if (it == buffers_.end()) {
    return nullptr;
  }
  return &it->second;
}

const memory::TensorBuffer* Session::find_buffer(const std::string& name) const {
  TensorId id = 0;
  if (!resolve_tensor(name, &id).ok()) {
    return nullptr;
  }
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

Error Session::set_tensor(const std::string& name_or_id,
                          const void* data,
                          std::size_t nbytes) {
  if (!ready_) {
    return Error::make(ErrorCode::InvalidArgument, "session not ready");
  }
  TensorId id = 0;
  Error err = resolve_tensor(name_or_id, &id);
  if (!err.ok()) {
    return err;
  }
  auto it = buffers_.find(id);
  if (it == buffers_.end()) {
    return Error::make(ErrorCode::NotFound, "buffer missing");
  }
  auto& buf = it->second;
  if (data == nullptr || nbytes != buf.nbytes) {
    return Error::make(ErrorCode::InvalidArgument,
                       "set_tensor size mismatch for " + name_or_id + " expected " +
                           std::to_string(buf.nbytes) + " got " +
                           std::to_string(nbytes));
  }
  std::memcpy(buf.data, data, nbytes);
  return Error::ok();
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
  if (!err.ok()) {
    return err;
  }
  const auto* buf = find_buffer(id);
  if (buf == nullptr) {
    return Error::make(ErrorCode::NotFound, "buffer missing");
  }
  if (data == nullptr || nbytes != buf->nbytes) {
    return Error::make(ErrorCode::InvalidArgument, "get_tensor size mismatch");
  }
  std::memcpy(data, buf->data, nbytes);
  return Error::ok();
}

Error Session::get_tensor_f32(const std::string& name_or_id,
                              std::vector<float>* values) const {
  if (values == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "values out null");
  }
  TensorId id = 0;
  Error err = resolve_tensor(name_or_id, &id);
  if (!err.ok()) {
    return err;
  }
  const auto* buf = find_buffer(id);
  if (buf == nullptr) {
    return Error::make(ErrorCode::NotFound, "buffer missing");
  }
  const std::size_t n = buf->nbytes / sizeof(float);
  values->resize(n);
  std::memcpy(values->data(), buf->data, buf->nbytes);
  return Error::ok();
}

Error Session::run() {
  if (!ready_) {
    return Error::make(ErrorCode::InvalidArgument, "session not ready");
  }

  std::vector<ScheduleDecision> decisions;
  Error err = scheduler_.schedule_plan(plan_, &decisions);
  if (!err.ok()) {
    return err;
  }
  (void)decisions;

  for (const auto& planned : plan_.ops) {
    const ir::Node* node = graph_.find_node(planned.node_id);
    if (node == nullptr) {
      return Error::make(ErrorCode::Internal,
                         "missing node " + std::to_string(planned.node_id));
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

    log::debug("session") << "dispatch " << node->op_name << " node=" << node->id;
    err = backend_->dispatch(node->op_name, node->op_version, inputs, &outputs,
                             node->attributes);
    if (!err.ok()) {
      return Error::make(err.code(),
                         "op '" + node->op_name + "' node=" +
                             std::to_string(node->id) + ": " + err.message());
    }
  }

  return backend_->synchronize();
}

std::string Session::debug_stats() const {
  std::ostringstream oss;
  oss << "graph=" << graph_.name << " ops=" << plan_.ops.size()
      << " tensors=" << buffers_.size();
  if (backend_) {
    oss << " backend=" << backend_->name();
    if (backend_->uses_host_fallback()) {
      oss << "(host-fallback)";
    }
  }
  if (allocator_) {
    oss << " " << allocator_->stats();
  }
  return oss.str();
}

}  // namespace runtime
}  // namespace uaii
