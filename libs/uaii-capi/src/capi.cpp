#include "uaii/c_api/uaii.h"

#include "uaii/ir/serialize.hpp"
#include "uaii/loaders/registry.hpp"
#include "uaii/profiler/profiler.hpp"
#include "uaii/runtime/memory_preset.hpp"
#include "uaii/runtime/session.hpp"
#include "uaii/version.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_error(const uaii::Error& err) {
  g_last_error = err.ok() ? std::string() : err.to_string();
}

void set_error_msg(const std::string& msg) {
  g_last_error = msg;
}

uaii_status to_status(uaii::ErrorCode code) {
  switch (code) {
    case uaii::ErrorCode::Ok: return UAII_OK;
    case uaii::ErrorCode::InvalidArgument: return UAII_ERR_INVALID_ARGUMENT;
    case uaii::ErrorCode::NotFound: return UAII_ERR_NOT_FOUND;
    case uaii::ErrorCode::IoError: return UAII_ERR_IO;
    case uaii::ErrorCode::ConfigError: return UAII_ERR_CONFIG;
    case uaii::ErrorCode::PluginError: return UAII_ERR_PLUGIN;
    case uaii::ErrorCode::AbiMismatch: return UAII_ERR_ABI;
    case uaii::ErrorCode::NotImplemented: return UAII_ERR_NOT_IMPLEMENTED;
    default: return UAII_ERR_INTERNAL;
  }
}

uaii_status from_error(const uaii::Error& err) {
  set_error(err);
  return to_status(err.code());
}

bool looks_like_ir(const std::string& path) {
  auto ends = [&](const char* suf) {
    const std::size_t n = std::strlen(suf);
    return path.size() >= n && path.compare(path.size() - n, n, suf) == 0;
  };
  return ends(".uaii.json") || ends(".json") || ends(".uaii");
}

struct SessionHandle {
  uaii::runtime::Session session;
};

}  // namespace

extern "C" {

void uaii_session_options_init(uaii_session_options* opts) {
  if (opts == nullptr) return;
  std::memset(opts, 0, sizeof(*opts));
  opts->struct_size = static_cast<uint32_t>(sizeof(uaii_session_options));
  opts->backend = "cpu";
  opts->weight_init = UAII_WEIGHT_INIT_NONE;  // fail closed
  opts->enable_fusion = 1;
  opts->enable_memory_reuse = 1;
  opts->enable_profiler = 0;
  opts->budget_bytes = 0;
  opts->enable_streaming = 0;
  opts->allow_missing_weights = 0;
  opts->weights_sandbox = nullptr;
  opts->compute_dtype = UAII_COMPUTE_DTYPE_F32;
  opts->keep_quantized_weights = 1;
  opts->max_context = 0;
  opts->kv_window = 0;
  opts->kv_sink = 4;
  opts->pin_layers = 0;
  opts->expert_cache_bytes = 0;
  opts->max_threads = 0;
  opts->single_staging = 0;
}

uaii_status uaii_get_version(int* major, int* minor, int* patch) {
  const uaii::Version v = uaii::version();
  if (major) *major = v.major;
  if (minor) *minor = v.minor;
  if (patch) *patch = v.patch;
  return UAII_OK;
}

const char* uaii_get_version_string(void) {
  return uaii::version_string();
}

void uaii_get_c_api_version(int* major, int* minor, int* patch) {
  if (major) *major = UAII_C_API_VERSION_MAJOR;
  if (minor) *minor = UAII_C_API_VERSION_MINOR;
  if (patch) *patch = UAII_C_API_VERSION_PATCH;
}

const char* uaii_get_c_api_version_string(void) {
  return UAII_C_API_VERSION_STRING;
}

const char* uaii_status_name(uaii_status status) {
  switch (status) {
    case UAII_OK: return "ok";
    case UAII_ERR_INVALID_ARGUMENT: return "invalid_argument";
    case UAII_ERR_NOT_FOUND: return "not_found";
    case UAII_ERR_IO: return "io";
    case UAII_ERR_CONFIG: return "config";
    case UAII_ERR_PLUGIN: return "plugin";
    case UAII_ERR_ABI: return "abi";
    case UAII_ERR_NOT_IMPLEMENTED: return "not_implemented";
    case UAII_ERR_INTERNAL: return "internal";
    default: return "unknown";
  }
}

const char* uaii_last_error(void) {
  return g_last_error.c_str();
}

uaii_status uaii_session_create(const char* path,
                                const uaii_session_options* opts,
                                uaii_session** out) {
  if (path == nullptr || out == nullptr) {
    set_error_msg("path/out null");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  *out = nullptr;

  uaii_session_options local;
  uaii_session_options_init(&local);
  if (opts != nullptr) {
    if (opts->struct_size == 0 || opts->struct_size > sizeof(uaii_session_options)) {
      set_error_msg("uaii_session_options.struct_size invalid");
      return UAII_ERR_ABI;
    }
    std::memcpy(&local, opts, static_cast<std::size_t>(opts->struct_size));
    local.struct_size = static_cast<uint32_t>(sizeof(uaii_session_options));
  }

  uaii::ir::Graph graph;
  uaii::Error err;
  const std::string p(path);
  if (looks_like_ir(p)) {
    err = uaii::ir::load_graph(p, &graph);
  } else {
    err = uaii::loaders::load_model(p, &graph);
  }
  if (!err.ok()) return from_error(err);

  auto handle = std::make_unique<SessionHandle>();
  uaii::runtime::SessionOptions sopts;
  sopts.backend_name = local.backend ? local.backend : "cpu";
  if (local.weights_dir) sopts.weights_dir = local.weights_dir;
  switch (local.weight_init) {
    case UAII_WEIGHT_INIT_ZEROS: sopts.weight_init = uaii::runtime::WeightInit::Zeros; break;
    case UAII_WEIGHT_INIT_ONES: sopts.weight_init = uaii::runtime::WeightInit::Ones; break;
    case UAII_WEIGHT_INIT_SEQUENCE: sopts.weight_init = uaii::runtime::WeightInit::Sequence; break;
    default: sopts.weight_init = uaii::runtime::WeightInit::None; break;
  }
  sopts.enable_fusion = local.enable_fusion != 0;
  sopts.enable_memory_reuse = local.enable_memory_reuse != 0;
  sopts.enable_profiler = local.enable_profiler != 0;
  if (local.profile_trace_path) sopts.profile_trace_path = local.profile_trace_path;
  sopts.allocator.budget_bytes = local.budget_bytes;
  sopts.enable_streaming = local.enable_streaming != 0;
  sopts.allow_missing_weights = local.allow_missing_weights != 0;
  if (local.weights_sandbox) sopts.weights_sandbox = local.weights_sandbox;
  if (local.compute_dtype == UAII_COMPUTE_DTYPE_F16) {
    sopts.compute_dtype = uaii::DType::F16;
  } else {
    sopts.compute_dtype = uaii::DType::F32;
  }
  sopts.keep_quantized_weights = local.keep_quantized_weights != 0;
  sopts.max_context = local.max_context;
  sopts.kv_window = local.kv_window;
  sopts.kv_sink = local.kv_sink;
  sopts.pin_layers = local.pin_layers;
  sopts.expert_cache_bytes = local.expert_cache_bytes;
  sopts.max_threads = local.max_threads;
  sopts.single_staging = local.single_staging != 0;

  err = handle->session.create(std::move(graph), sopts);
  if (!err.ok()) return from_error(err);

  *out = reinterpret_cast<uaii_session*>(handle.release());
  g_last_error.clear();
  return UAII_OK;
}

void uaii_session_destroy(uaii_session* session) {
  delete reinterpret_cast<SessionHandle*>(session);
}

uaii_status uaii_session_set_f32(uaii_session* session,
                                 const char* tensor_name,
                                 const float* data,
                                 size_t n) {
  if (session == nullptr || tensor_name == nullptr || data == nullptr) {
    set_error_msg("set_f32 null arg");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  auto* h = reinterpret_cast<SessionHandle*>(session);
  std::vector<float> values(data, data + n);
  return from_error(h->session.set_tensor_f32(tensor_name, values));
}

uaii_status uaii_session_get_f32(uaii_session* session,
                                 const char* tensor_name,
                                 float* out,
                                 size_t capacity,
                                 size_t* out_n) {
  if (session == nullptr || tensor_name == nullptr || out_n == nullptr) {
    set_error_msg("get_f32 null arg");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  auto* h = reinterpret_cast<SessionHandle*>(session);
  std::vector<float> values;
  uaii::Error err = h->session.get_tensor_f32(tensor_name, &values);
  if (!err.ok()) return from_error(err);
  *out_n = values.size();
  if (out == nullptr) return UAII_OK;
  if (capacity < values.size()) {
    set_error_msg("get_f32 buffer too small");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  std::memcpy(out, values.data(), values.size() * sizeof(float));
  g_last_error.clear();
  return UAII_OK;
}

uaii_status uaii_session_run(uaii_session* session) {
  if (session == nullptr) {
    set_error_msg("session null");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  auto* h = reinterpret_cast<SessionHandle*>(session);
  return from_error(h->session.run());
}

uaii_status uaii_session_generate(uaii_session* session,
                                  const int64_t* prompt_tokens,
                                  size_t prompt_n,
                                  int64_t max_new_tokens,
                                  int64_t* out_tokens,
                                  size_t capacity,
                                  size_t* out_n) {
  if (session == nullptr || out_tokens == nullptr || out_n == nullptr) {
    set_error_msg("generate null args");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  auto* h = reinterpret_cast<SessionHandle*>(session);
  std::vector<std::int64_t> prompt;
  if (prompt_tokens != nullptr && prompt_n > 0) {
    prompt.assign(prompt_tokens, prompt_tokens + prompt_n);
  }
  std::vector<std::int64_t> out;
  uaii::Error err = h->session.generate(prompt, max_new_tokens, &out);
  if (!err.ok()) return from_error(err);
  const size_t n = std::min(capacity, out.size());
  for (size_t i = 0; i < n; ++i) out_tokens[i] = out[i];
  *out_n = n;
  g_last_error.clear();
  return UAII_OK;
}

uaii_status uaii_session_profile_summary(uaii_session* session,
                                         char* buf,
                                         size_t capacity) {
  if (session == nullptr || buf == nullptr || capacity == 0) {
    set_error_msg("profile_summary null/empty");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  auto* h = reinterpret_cast<SessionHandle*>(session);
  const std::string s = h->session.profiler().summary();
  if (s.size() + 1 > capacity) {
    set_error_msg("profile_summary buffer too small");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  std::memcpy(buf, s.c_str(), s.size() + 1);
  g_last_error.clear();
  return UAII_OK;
}

uaii_status uaii_session_write_trace(uaii_session* session, const char* path) {
  if (session == nullptr || path == nullptr) {
    set_error_msg("write_trace null");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  auto* h = reinterpret_cast<SessionHandle*>(session);
  return from_error(uaii::profiler::write_chrome_trace(h->session.profiler(), path));
}

uaii_status uaii_resolve_memory_preset(const char* name, uint64_t ram_gb_override,
                                     uaii_memory_preset_info* out) {
  if (out == nullptr) {
    set_error_msg("memory preset out null");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  if (out->struct_size == 0 || out->struct_size > sizeof(uaii_memory_preset_info)) {
    set_error_msg("uaii_memory_preset_info.struct_size invalid");
    return UAII_ERR_ABI;
  }
  uaii::runtime::MemoryPreset preset = uaii::runtime::MemoryPreset::None;
  if (!uaii::runtime::parse_memory_preset(name ? name : "", &preset)) {
    set_error_msg("unknown memory preset");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  uaii::runtime::MemoryPresetConfig cfg;
  uaii::Error err = uaii::runtime::resolve_memory_preset(preset, ram_gb_override, &cfg);
  if (!err.ok()) return from_error(err);

  uaii_memory_preset_info full;
  std::memset(&full, 0, sizeof(full));
  full.struct_size = static_cast<uint32_t>(sizeof(full));
  full.ram_budget_bytes = cfg.ram_budget_bytes;
  full.pin_layers = cfg.pin_layers;
  full.expert_cache_bytes = cfg.expert_cache_bytes;
  full.kv_window = cfg.kv_window;
  full.kv_sink = cfg.kv_sink;
  full.enable_streaming = cfg.enable_streaming ? 1 : 0;
  full.single_staging = cfg.single_staging ? 1 : 0;
  full.max_threads = cfg.max_threads;
  full.budget_clamped = cfg.budget_clamped ? 1 : 0;
  full.available_bytes = uaii::runtime::system_available_bytes();
  std::snprintf(full.label, sizeof(full.label), "%s", cfg.label.c_str());

  const uint32_t nbytes = out->struct_size;
  std::memcpy(out, &full, static_cast<std::size_t>(nbytes));
  out->struct_size = nbytes;
  g_last_error.clear();
  return UAII_OK;
}

void uaii_session_options_apply_preset(uaii_session_options* opts,
                                       const uaii_memory_preset_info* preset) {
  if (opts == nullptr || preset == nullptr) return;
  if (preset->struct_size < offsetof(uaii_memory_preset_info, label)) return;
  opts->budget_bytes = preset->ram_budget_bytes;
  opts->enable_streaming = preset->enable_streaming;
  opts->kv_window = preset->kv_window;
  opts->kv_sink = preset->kv_sink;
  opts->pin_layers = preset->pin_layers;
  opts->expert_cache_bytes = preset->expert_cache_bytes;
  opts->max_threads = preset->max_threads;
  opts->single_staging = preset->single_staging;
}

uint64_t uaii_system_available_bytes(void) {
  return uaii::runtime::system_available_bytes();
}

uint64_t uaii_process_rss_bytes(void) {
  return uaii::runtime::process_current_rss_bytes();
}

void uaii_session_request_cancel(uaii_session* session) {
  if (session == nullptr) return;
  reinterpret_cast<SessionHandle*>(session)->session.request_cancel();
}

void uaii_session_set_background(uaii_session* session, int background) {
  if (session == nullptr) return;
  reinterpret_cast<SessionHandle*>(session)->session.set_background(background != 0);
}

void uaii_session_memory_warning(uaii_session* session) {
  if (session == nullptr) return;
  reinterpret_cast<SessionHandle*>(session)->session.on_memory_warning();
}

uaii_status uaii_convert_model(const char* input_path, const char* output_path) {
  if (input_path == nullptr || output_path == nullptr) {
    set_error_msg("convert null path");
    return UAII_ERR_INVALID_ARGUMENT;
  }
  return from_error(uaii::loaders::convert_model(input_path, output_path));
}

}  // extern "C"
