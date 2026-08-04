#pragma once

#include "uaii/c_api/plugin_abi.h"
#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <memory>
#include <string>
#include <vector>

namespace uaii {

[[nodiscard]] UAII_API const char* to_string(uaii_plugin_kind kind) noexcept;

/// Metadata for a discovered / loaded plugin.
struct PluginDescriptor {
  std::string path;
  uint32_t abi_version = 0;
  uaii_plugin_kind kind = UAII_PLUGIN_KIND_UNKNOWN;
  std::string name;
  std::string version;
  std::string description;
  bool loaded = false;
};

/// Owns a loaded dynamic library and its lifecycle hooks.
class UAII_API Plugin {
 public:
  Plugin() = default;
  ~Plugin();

  Plugin(const Plugin&) = delete;
  Plugin& operator=(const Plugin&) = delete;
  Plugin(Plugin&&) noexcept;
  Plugin& operator=(Plugin&&) noexcept;

  [[nodiscard]] Error load(const std::string& path);
  void unload() noexcept;

  [[nodiscard]] bool is_loaded() const noexcept { return handle_ != nullptr; }
  [[nodiscard]] const PluginDescriptor& descriptor() const noexcept { return descriptor_; }

 private:
  void* handle_ = nullptr;
  uaii_plugin_init_fn init_fn_ = nullptr;
  uaii_plugin_shutdown_fn shutdown_fn_ = nullptr;
  PluginDescriptor descriptor_;
};

class UAII_API PluginRegistry {
 public:
  /// Scan directories for shared libraries (.dll / .so / .dylib).
  [[nodiscard]] Error discover(const std::vector<std::string>& directories);

  /// Load all discovered plugins (ABI-compatible only).
  [[nodiscard]] Error load_all();

  /// Load a single plugin by filesystem path.
  [[nodiscard]] Error load_path(const std::string& path);

  [[nodiscard]] const std::vector<PluginDescriptor>& discovered() const noexcept {
    return discovered_;
  }

  [[nodiscard]] std::size_t loaded_count() const noexcept { return plugins_.size(); }

  [[nodiscard]] const std::vector<std::unique_ptr<Plugin>>& plugins() const noexcept {
    return plugins_;
  }

  void clear() noexcept;

 private:
  std::vector<PluginDescriptor> discovered_;
  std::vector<std::unique_ptr<Plugin>> plugins_;
};

}  // namespace uaii
