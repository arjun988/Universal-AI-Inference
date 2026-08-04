#include "uaii/core/plugin.hpp"

#include "uaii/core/log.hpp"

#include <filesystem>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace uaii {
namespace fs = std::filesystem;

namespace {

void* load_library(const std::string& path) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_library(void* handle) {
  if (handle == nullptr) {
    return;
  }
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

void* lookup_symbol(void* handle, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

std::string last_dynload_error() {
#if defined(_WIN32)
  const DWORD code = GetLastError();
  return "Win32 error " + std::to_string(static_cast<unsigned long>(code));
#else
  const char* err = dlerror();
  return err != nullptr ? std::string(err) : std::string("unknown dlerror");
#endif
}

bool has_plugin_extension(const fs::path& path) {
  const auto ext = path.extension().string();
#if defined(_WIN32)
  return ext == ".dll";
#elif defined(__APPLE__)
  return ext == ".dylib" || ext == ".so";
#else
  return ext == ".so";
#endif
}

PluginDescriptor make_descriptor_from_info(const std::string& path,
                                           const uaii_plugin_info* info) {
  PluginDescriptor d;
  d.path = path;
  if (info == nullptr) {
    return d;
  }
  d.abi_version = info->abi_version;
  d.kind = info->kind;
  d.name = info->name;
  d.version = info->version;
  d.description = info->description;
  return d;
}

}  // namespace

const char* to_string(uaii_plugin_kind kind) noexcept {
  switch (kind) {
    case UAII_PLUGIN_KIND_LOADER: return "loader";
    case UAII_PLUGIN_KIND_OPERATOR: return "operator";
    case UAII_PLUGIN_KIND_BACKEND: return "backend";
    case UAII_PLUGIN_KIND_STORAGE: return "storage";
    case UAII_PLUGIN_KIND_SCHEDULER: return "scheduler";
    case UAII_PLUGIN_KIND_TOKENIZER: return "tokenizer";
    case UAII_PLUGIN_KIND_QUANTIZATION: return "quantization";
    case UAII_PLUGIN_KIND_PROFILER: return "profiler";
    case UAII_PLUGIN_KIND_OPTIMIZER: return "optimizer";
    case UAII_PLUGIN_KIND_PROBE: return "probe";
    default: return "unknown";
  }
}

Plugin::~Plugin() {
  unload();
}

Plugin::Plugin(Plugin&& other) noexcept {
  *this = std::move(other);
}

Plugin& Plugin::operator=(Plugin&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  unload();
  handle_ = other.handle_;
  init_fn_ = other.init_fn_;
  shutdown_fn_ = other.shutdown_fn_;
  descriptor_ = std::move(other.descriptor_);
  other.handle_ = nullptr;
  other.init_fn_ = nullptr;
  other.shutdown_fn_ = nullptr;
  return *this;
}

Error Plugin::load(const std::string& path) {
  unload();

  void* handle = load_library(path);
  if (handle == nullptr) {
    return Error::make(ErrorCode::PluginError,
                       "failed to load plugin '" + path + "': " + last_dynload_error());
  }

  auto get_info = reinterpret_cast<uaii_plugin_get_info_fn>(
      lookup_symbol(handle, "uaii_plugin_get_info"));
  auto init_fn = reinterpret_cast<uaii_plugin_init_fn>(
      lookup_symbol(handle, "uaii_plugin_init"));
  auto shutdown_fn = reinterpret_cast<uaii_plugin_shutdown_fn>(
      lookup_symbol(handle, "uaii_plugin_shutdown"));

  if (get_info == nullptr || init_fn == nullptr || shutdown_fn == nullptr) {
    close_library(handle);
    return Error::make(ErrorCode::PluginError,
                       "plugin '" + path +
                           "' missing required exports "
                           "(uaii_plugin_get_info/init/shutdown)");
  }

  const uaii_plugin_info* info = get_info();
  if (info == nullptr) {
    close_library(handle);
    return Error::make(ErrorCode::PluginError,
                       "plugin '" + path + "' returned null info");
  }

  if (info->abi_version != UAII_PLUGIN_ABI_VERSION) {
    close_library(handle);
    return Error::make(ErrorCode::AbiMismatch,
                       "plugin '" + path + "' ABI " +
                           std::to_string(info->abi_version) +
                           " incompatible with host ABI " +
                           std::to_string(UAII_PLUGIN_ABI_VERSION));
  }

  const int rc = init_fn();
  if (rc != 0) {
    close_library(handle);
    return Error::make(ErrorCode::PluginError,
                       "plugin '" + path + "' init failed with code " +
                           std::to_string(rc));
  }

  handle_ = handle;
  init_fn_ = init_fn;
  shutdown_fn_ = shutdown_fn;
  descriptor_ = make_descriptor_from_info(path, info);
  descriptor_.loaded = true;

  log::info("plugin") << "loaded " << descriptor_.name << " v"
                      << descriptor_.version << " (" << to_string(descriptor_.kind)
                      << ") from " << path;
  return Error::ok();
}

void Plugin::unload() noexcept {
  if (handle_ == nullptr) {
    return;
  }
  if (shutdown_fn_ != nullptr) {
    shutdown_fn_();
  }
  close_library(handle_);
  handle_ = nullptr;
  init_fn_ = nullptr;
  shutdown_fn_ = nullptr;
  descriptor_.loaded = false;
}

Error PluginRegistry::discover(const std::vector<std::string>& directories) {
  discovered_.clear();

  for (const auto& dir : directories) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
      log::debug("plugin") << "skip missing plugin dir: " << dir;
      continue;
    }

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
      if (ec) {
        return Error::make(ErrorCode::IoError,
                           "failed to iterate plugin dir '" + dir + "': " +
                               ec.message());
      }
      if (!entry.is_regular_file(ec)) {
        continue;
      }
      const auto path = entry.path();
      if (!has_plugin_extension(path)) {
        continue;
      }

      // Probe metadata without keeping the library resident.
      void* handle = load_library(path.string());
      if (handle == nullptr) {
        log::warn("plugin") << "discover: cannot open " << path.string() << ": "
                            << last_dynload_error();
        continue;
      }

      auto get_info = reinterpret_cast<uaii_plugin_get_info_fn>(
          lookup_symbol(handle, "uaii_plugin_get_info"));
      PluginDescriptor desc;
      desc.path = path.string();
      if (get_info != nullptr) {
        desc = make_descriptor_from_info(path.string(), get_info());
      } else {
        desc.name = path.stem().string();
        desc.description = "missing uaii_plugin_get_info";
      }
      close_library(handle);
      discovered_.push_back(std::move(desc));
    }
  }

  log::info("plugin") << "discovered " << discovered_.size() << " plugin candidate(s)";
  return Error::ok();
}

Error PluginRegistry::load_all() {
  for (const auto& desc : discovered_) {
    Error err = load_path(desc.path);
    if (!err.ok()) {
      log::warn("plugin") << err.to_string();
      // Continue loading others; doctor reports failures.
    }
  }
  return Error::ok();
}

Error PluginRegistry::load_path(const std::string& path) {
  auto plugin = std::make_unique<Plugin>();
  Error err = plugin->load(path);
  if (!err.ok()) {
    return err;
  }
  plugins_.push_back(std::move(plugin));
  return Error::ok();
}

void PluginRegistry::clear() noexcept {
  plugins_.clear();
  discovered_.clear();
}

}  // namespace uaii
