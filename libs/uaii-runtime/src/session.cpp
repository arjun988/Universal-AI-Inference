#include "uaii/runtime/session.hpp"

#include "uaii/backends/factory.hpp"
#include "uaii/backends/host_executable.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/dtype.hpp"
#include "uaii/ir/registry.hpp"
#include "uaii/ir/validator.hpp"
#include "uaii/kernels/gemm.hpp"
#include "uaii/kernels/kernels.hpp"
#include "uaii/loaders/registry.hpp"
#include "uaii/quant/quantizer.hpp"

#include <algorithm>
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

std::int64_t meta_i64(const ir::Graph& g, const char* key, std::int64_t def) {
  auto it = g.metadata.find(key);
  if (it == g.metadata.end() || it->second.empty()) return def;
  try {
    return std::stoll(it->second);
  } catch (...) {
    return def;
  }
}

struct ActiveKvBinding {
  explicit ActiveKvBinding(KvCache* cache) { kernels::set_active_kv_cache(cache); }
  ~ActiveKvBinding() { kernels::set_active_kv_cache(nullptr); }
  ActiveKvBinding(const ActiveKvBinding&) = delete;
  ActiveKvBinding& operator=(const ActiveKvBinding&) = delete;
};

}  // namespace

Session::Session() = default;

Session::~Session() {
  destroy();
}

Error Session::create(ir::Graph graph, SessionOptions options) {
  destroy();
  options_ = std::move(options);

  if (options_.compute_dtype != DType::F32 && options_.compute_dtype != DType::F16) {
    return Error::make(ErrorCode::InvalidArgument,
                       "compute_dtype must be F32 or F16");
  }

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
  scheduler_.set_preferred(backend_->device_type());
  scheduler_.set_attention_host_fallback(backend_->attention_host_fallback());

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
    const BackendCapabilities caps = backend_->capabilities();
    stream_async_h2d_ =
        caps.native_available && caps.supports_async &&
        backend_->device_type() != DeviceType::Cpu;
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
                       << " compute_dtype=" << to_string(options_.compute_dtype)
                       << " keep_quantized_weights="
                       << (options_.keep_quantized_weights ? "true" : "false")
                       << " gemm=" << kernels::GemmRegistry::instance().describe()
                       << " ops=" << plan_.ops.size() << " " << report_.summary
                       << " " << allocator_->stats();
  return Error::success();
}

void Session::destroy() noexcept {
  kernels::set_active_kv_cache(nullptr);
  kv_.reset();
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
  for (auto& kv : stream_device_) {
    if (backend_ && kv.second.device_ptr != nullptr) {
      (void)backend_->free(kv.second.device_ptr);
    }
  }
  stream_device_.clear();
  stream_async_h2d_ = false;
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
    if (t.is_weight && options_.keep_quantized_weights &&
        quant::is_gguf_block_quant(t.quant_format)) {
      const std::size_t n = ir::shape_numel(t.shape);
      buf.id = t.id;
      buf.dtype = DType::Unknown;
      buf.shape = t.shape;
      buf.quant_format = t.quant_format;
      buf.quant_rows = t.shape.dims.empty() ? 0 : t.shape.dims[0];
      buf.quant_cols = t.shape.dims.size() > 1 ? t.shape.dims[1] : 0;
      buf.nbytes = quant::packed_nbytes(t.quant_format, n);
      buf.owned = true;
      Error err = allocator_->allocate_bytes(buf.nbytes, &buf.data);
      if (!err.ok()) return err;
      std::memset(buf.data, 0, buf.nbytes);
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
    float* data = quant::is_gguf_block_quant(buf.quant_format) ? nullptr : memory::as_f32(buf);
    const std::size_t n = quant::is_gguf_block_quant(buf.quant_format)
                              ? ir::shape_numel(t.shape)
                              : buf.nbytes / sizeof(float);

    bool loaded = false;
    Error load_err = Error::success();
    if (!t.weight_ref.empty()) {
      if (options_.keep_quantized_weights && quant::is_gguf_block_quant(t.quant_format)) {
        std::vector<std::uint8_t> packed;
        quant::QuantFormat fmt = t.quant_format;
        load_err = loaders::load_weight_ref_auto(t.weight_ref, options_.weights_dir, t.shape,
                                                 true, &packed, &fmt, nullptr, 0);
        if (load_err.ok() && packed.size() == buf.nbytes) {
          std::memcpy(buf.data, packed.data(), packed.size());
          buf.quant_format = fmt;
          loaded = true;
        }
      }
      if (!loaded && data != nullptr) {
        load_err = loaders::load_weight_ref_f32(t.weight_ref, options_.weights_dir, t.shape,
                                                data, buf.nbytes);
        if (load_err.ok()) {
          loaded = true;
          buf.quant_format = quant::QuantFormat::F32;
        }
      }
      if (!loaded) {
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

    if (data == nullptr) {
      return Error::make(ErrorCode::NotFound,
                         "failed to load packed weight '" + t.name + "'");
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
  v.quant_format = buf.quant_format;
  v.quant_rows = buf.quant_rows;
  v.quant_cols = buf.quant_cols;
  return v;
}

Error Session::ensure_stream_device(TensorId tid, std::size_t nbytes) {
  if (backend_ == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "no backend for stream device");
  }
  auto& slot = stream_device_[tid];
  if (slot.device_ptr != nullptr && slot.nbytes >= nbytes) {
    return Error::success();
  }
  if (slot.device_ptr != nullptr) {
    (void)backend_->free(slot.device_ptr);
    slot.device_ptr = nullptr;
    slot.nbytes = 0;
    slot.h2d_in_flight = false;
  }
  Error err = backend_->allocate(nbytes, &slot.device_ptr);
  if (!err.ok()) return err;
  slot.nbytes = nbytes;
  slot.h2d_in_flight = false;
  return Error::success();
}

Error Session::upload_streamed_weight(TensorId tid, const void* host, std::size_t nbytes) {
  auto it = buffers_.find(tid);
  if (it == buffers_.end()) {
    return Error::make(ErrorCode::Internal, "missing streamed buffer");
  }
  if (!stream_async_h2d_ || host == nullptr || nbytes == 0) {
    it->second.data = const_cast<void*>(host);
    it->second.nbytes = nbytes;
    return Error::success();
  }
  Error err = ensure_stream_device(tid, nbytes);
  if (!err.ok()) return err;
  auto& dev = stream_device_[tid];
  err = backend_->copy_h2d_async(host, dev.device_ptr, nbytes);
  if (err.code() == ErrorCode::NotImplemented) {
    err = backend_->copy_h2d(host, dev.device_ptr, nbytes);
    dev.h2d_in_flight = false;
  } else if (err.ok()) {
    dev.h2d_in_flight = true;
  }
  if (!err.ok()) return err;
  it->second.data = dev.device_ptr;
  it->second.nbytes = nbytes;
  return Error::success();
}

Error Session::sync_stream_h2d(TensorId tid) {
  if (!stream_async_h2d_ || backend_ == nullptr) {
    return Error::success();
  }
  auto it = stream_device_.find(tid);
  if (it == stream_device_.end() || !it->second.h2d_in_flight) {
    return Error::success();
  }
  Error err = backend_->synchronize();
  if (!err.ok()) return err;
  it->second.h2d_in_flight = false;
  return Error::success();
}

void Session::try_prefetch_stream_h2d(TensorId tid) {
  if (!stream_async_h2d_ || !streaming_ || !streaming_->is_streamed(tid)) {
    return;
  }
  const void* host = nullptr;
  std::size_t nbytes = 0;
  if (!streaming_->try_peek_prefetched(tid, &host, &nbytes)) {
    return;
  }
  auto dit = stream_device_.find(tid);
  if (dit != stream_device_.end() && dit->second.h2d_in_flight) {
    return;
  }
  if (!ensure_stream_device(tid, nbytes).ok()) {
    return;
  }
  auto& dev = stream_device_[tid];
  Error err = backend_->copy_h2d_async(host, dev.device_ptr, nbytes);
  if (err.code() == ErrorCode::NotImplemented) {
    err = backend_->copy_h2d(host, dev.device_ptr, nbytes);
    dev.h2d_in_flight = false;
  } else if (err.ok()) {
    dev.h2d_in_flight = true;
  }
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
    err = upload_streamed_weight(tid, data, nbytes);
    if (!err.ok()) return err;
    err = sync_stream_h2d(tid);
    if (!err.ok()) return err;
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

Error Session::ensure_kv_cache(std::int64_t max_seq) {
  std::int64_t n_layers = meta_i64(graph_, "n_layers", 0);
  if (n_layers <= 0) {
    for (const auto& n : graph_.nodes) {
      if (n.op_name == "Attention") ++n_layers;
    }
  }
  std::int64_t n_heads = meta_i64(graph_, "n_heads", 0);
  if (n_heads <= 0) {
    for (const auto& n : graph_.nodes) {
      if (n.op_name != "Attention") continue;
      for (const auto& a : n.attributes) {
        if (a.key == "num_heads" && a.type == ir::AttributeType::Int) {
          n_heads = std::get<std::int64_t>(a.value);
          break;
        }
      }
      if (n_heads > 0) break;
    }
  }
  if (n_heads <= 0) n_heads = 1;

  std::int64_t dim = meta_i64(graph_, "embedding_length", 0);
  if (dim <= 0) dim = meta_i64(graph_, "dim", 0);
  if (dim <= 0) {
    for (const auto& t : graph_.tensors) {
      if (t.name.find("emb") != std::string::npos && t.shape.dims.size() == 2) {
        dim = t.shape.dims[1];
        break;
      }
    }
  }
  if (dim <= 0) {
    for (const auto& t : graph_.tensors) {
      if (t.name == "hidden" && t.shape.dims.size() >= 2) {
        dim = t.shape.dims.back();
        break;
      }
    }
  }
  if (dim <= 0) {
    // Fall back to Attention activation tensors (…/q, …/k, …/attn).
    for (const auto& t : graph_.tensors) {
      if (!t.is_weight && t.shape.dims.size() >= 2) {
        const auto& n = t.name;
        if (n.size() >= 2 &&
            (n.compare(n.size() - 2, 2, ".q") == 0 || n == "q" ||
             n.compare(n.size() - 5, 5, ".attn") == 0)) {
          dim = t.shape.dims.back();
          break;
        }
      }
    }
  }
  if (n_layers <= 0 || dim <= 0 || max_seq <= 0) {
    // No transformer stack — leave kv_ unset; Attention falls back to no-cache path.
    kv_.reset();
    return Error::success();
  }
  if (!kv_) kv_ = std::make_unique<KvCache>();
  Error err = kv_->configure(n_layers, /*batch=*/1, max_seq, dim, n_heads);
  if (!err.ok()) return err;
  kv_->reset();
  return Error::success();
}

Error Session::write_tokens_step(TensorId tokens_id, std::int64_t token) {
  auto it = buffers_.find(tokens_id);
  if (it == buffers_.end() || it->second.data == nullptr) {
    return Error::make(ErrorCode::NotFound, "tokens buffer missing");
  }
  auto& buf = it->second;
  const std::size_t n = buf.nbytes / sizeof(float);
  if (n == 0) {
    return Error::make(ErrorCode::InvalidArgument, "tokens buffer empty");
  }
  float* data = memory::as_f32(buf);
  // seq=1 graphs: single id. Longer buffers: write id at the last sequence slot.
  std::fill(data, data + n, 0.f);
  data[n - 1] = static_cast<float>(token);
  return Error::success();
}

Error Session::read_argmax_token(TensorId scores_id, std::int64_t* token) const {
  SampleParams greedy;
  std::mt19937_64 rng{0};
  return read_sample_token(scores_id, greedy, {}, &rng, token);
}

Error Session::read_sample_token(TensorId scores_id,
                                 const SampleParams& sample,
                                 const std::vector<std::int64_t>& history,
                                 std::mt19937_64* rng,
                                 std::int64_t* token) const {
  if (token == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "token out null");
  }
  const auto* buf = find_buffer(scores_id);
  if (buf == nullptr || buf->data == nullptr) {
    return Error::make(ErrorCode::NotFound, "scores buffer missing");
  }
  const float* data = memory::as_f32(*buf);
  const std::size_t n = buf->nbytes / sizeof(float);
  if (n == 0) {
    return Error::make(ErrorCode::InvalidArgument, "scores empty");
  }
  std::size_t row_start = 0;
  std::size_t row_n = n;
  if (buf->shape.dims.size() >= 2) {
    const std::int64_t vocab = buf->shape.dims.back();
    if (vocab > 0 && n % static_cast<std::size_t>(vocab) == 0) {
      row_n = static_cast<std::size_t>(vocab);
      row_start = n - row_n;
    }
  }
  return sample_token_f32(data + row_start, row_n, sample, history, rng, token);
}

Error Session::generate(const std::vector<std::int64_t>& prompt_tokens,
                        std::int64_t max_new_tokens,
                        std::vector<std::int64_t>* out_tokens,
                        const std::vector<std::int64_t>& stop_token_ids,
                        const OnNewToken& on_new_token,
                        const SampleParams& sample) {
  if (!ready_) {
    return Error::make(ErrorCode::InvalidArgument, "session not ready");
  }
  if (out_tokens == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "out_tokens null");
  }
  if (prompt_tokens.empty()) {
    return Error::make(ErrorCode::InvalidArgument, "prompt_tokens empty");
  }
  if (max_new_tokens < 0) {
    return Error::make(ErrorCode::InvalidArgument, "max_new_tokens < 0");
  }

  auto is_stop = [&](std::int64_t tok) {
    for (const auto s : stop_token_ids) {
      if (s == tok) return true;
    }
    return false;
  };

  std::int64_t max_ctx = options_.max_context;
  if (max_ctx <= 0) {
    max_ctx = meta_i64(graph_, "max_context", 0);
  }
  if (max_ctx <= 0) {
    max_ctx = meta_i64(graph_, "context_length", 0);
  }
  if (max_ctx <= 0) {
    max_ctx = meta_i64(graph_, "llama.context_length", 0);
  }

  const std::int64_t prompt_len = static_cast<std::int64_t>(prompt_tokens.size());
  const std::int64_t total = prompt_len + max_new_tokens;
  if (max_ctx > 0) {
    if (prompt_len > max_ctx) {
      return Error::make(ErrorCode::InvalidArgument,
                         "prompt length exceeds max_context (" +
                             std::to_string(prompt_len) + " > " +
                             std::to_string(max_ctx) + ")");
    }
    if (total > max_ctx) {
      return Error::make(ErrorCode::InvalidArgument,
                         "prompt+max_new_tokens exceeds max_context (" +
                             std::to_string(total) + " > " + std::to_string(max_ctx) +
                             ")");
    }
  }

  // Resolve tokens input + probs/logits output.
  TensorId tokens_id = 0;
  if (name_to_id_.count("tokens")) {
    tokens_id = name_to_id_["tokens"];
  } else if (!graph_.inputs.empty()) {
    tokens_id = graph_.inputs.front();
  } else {
    return Error::make(ErrorCode::NotFound, "generate: no tokens input");
  }

  // Prefer logits when sampling (temperature/top-p need pre-softmax scores).
  TensorId scores_id = 0;
  const bool want_logits = !sample.greedy();
  if (want_logits && name_to_id_.count("logits")) {
    scores_id = name_to_id_["logits"];
  } else if (name_to_id_.count("probs")) {
    scores_id = name_to_id_["probs"];
  } else if (name_to_id_.count("logits")) {
    scores_id = name_to_id_["logits"];
  } else if (!graph_.outputs.empty()) {
    scores_id = graph_.outputs.front();
  } else {
    return Error::make(ErrorCode::NotFound, "generate: no probs/logits output");
  }

  std::mt19937_64 rng;
  if (sample.has_seed) {
    rng.seed(sample.seed);
  } else {
    std::random_device rd;
    rng.seed((static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd()));
  }

  const std::int64_t kv_max =
      max_ctx > 0 ? max_ctx
                  : std::max<std::int64_t>(total, prompt_len + std::max<std::int64_t>(max_new_tokens, 1));
  Error err = ensure_kv_cache(kv_max);
  if (!err.ok()) return err;

  out_tokens->clear();
  out_tokens->reserve(static_cast<std::size_t>(total));

  // Prefill: feed each prompt token (seq=1 graphs) so KV grows when attrs enable it.
  for (std::int64_t i = 0; i < prompt_len; ++i) {
    if (max_ctx > 0 && (i + 1) > max_ctx) {
      return Error::make(ErrorCode::InvalidArgument, "context length exceeded during prefill");
    }
    err = write_tokens_step(tokens_id, prompt_tokens[static_cast<std::size_t>(i)]);
    if (!err.ok()) return err;
    err = run();
    if (!err.ok()) return err;
    out_tokens->push_back(prompt_tokens[static_cast<std::size_t>(i)]);
  }

  // Decode: sample next token (greedy when SampleParams::greedy()).
  for (std::int64_t n = 0; n < max_new_tokens; ++n) {
    const std::int64_t pos = static_cast<std::int64_t>(out_tokens->size());
    if (max_ctx > 0 && pos >= max_ctx) {
      return Error::make(ErrorCode::InvalidArgument, "context length exceeded during decode");
    }
    std::int64_t next = 0;
    err = read_sample_token(scores_id, sample, *out_tokens, &rng, &next);
    if (!err.ok()) return err;
    out_tokens->push_back(next);
    if (on_new_token) {
      if (!on_new_token(next)) {
        return Error::success();
      }
    }
    if (is_stop(next)) {
      return Error::success();
    }
    if (n + 1 >= max_new_tokens) break;
    if (max_ctx > 0 && static_cast<std::int64_t>(out_tokens->size()) >= max_ctx) {
      return Error::make(ErrorCode::InvalidArgument, "context length exceeded during decode");
    }
    err = write_tokens_step(tokens_id, next);
    if (!err.ok()) return err;
    err = run();
    if (!err.ok()) return err;
  }
  return Error::success();
}

Error Session::run() {
  if (!ready_) {
    return Error::make(ErrorCode::InvalidArgument, "session not ready");
  }

  ActiveKvBinding kv_bind(kv_.get());

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
          if (streaming_->is_streamed(tid)) {
            streaming_->prefetch(tid);
            try_prefetch_stream_h2d(tid);
          }
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
      const ScheduleDecision* sched = oi < decisions.size() ? &decisions[oi] : nullptr;
      const bool host_path =
          sched != nullptr && sched->device == DeviceType::Cpu &&
          backend_->device_type() != DeviceType::Cpu;
      std::string prof_args = "node=" + std::to_string(node->id);
      if (sched != nullptr && !sched->reason.empty()) {
        prof_args += std::string(" device=") + to_string(sched->device) +
                     " reason=" + sched->reason;
      }
      profiler::Profiler::Scope scope(
          options_.enable_profiler ? &profiler_ : nullptr, node->op_name,
          profiler::EventCategory::Kernel, prof_args);
      if (host_path) {
        auto* host_be = dynamic_cast<backends::HostExecutableBackend*>(backend_.get());
        if (host_be != nullptr) {
          err = host_be->dispatch_on_host_path(node->op_name, node->op_version, inputs,
                                               &outputs, node->attributes);
        } else {
          err = backend_->dispatch(node->op_name, node->op_version, inputs, &outputs,
                                   node->attributes);
        }
      } else {
        err = backend_->dispatch(node->op_name, node->op_version, inputs, &outputs,
                                 node->attributes);
      }
    }
    if (!err.ok()) {
      return Error::make(err.code(),
                         "op '" + node->op_name + "' node=" +
                             std::to_string(node->id) + ": " + err.message());
    }
  }

  err = backend_->synchronize();
  if (!err.ok()) return err;

  if (kv_ && kv_->configured()) {
    kv_->commit_step();
  }

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
