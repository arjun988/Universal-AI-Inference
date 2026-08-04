#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {

/// Hierarchical configuration: section.key -> string value.
/// Supports a TOML-like subset (sections, keys, strings, bools, ints)
/// and environment overlays with prefix UAII_.
class UAII_API Config {
 public:
  Config() = default;

  /// Load from a TOML-subset file. Missing file returns NotFound.
  [[nodiscard]] Error load_file(const std::string& path);

  /// Parse TOML-subset text in memory.
  [[nodiscard]] Error parse(const std::string& text, const std::string& source_name = "<memory>");

  /// Apply environment variables: UAII_LOG_LEVEL -> log.level, etc.
  /// Nested keys use double underscore: UAII_PLUGIN__DIRS -> plugin.dirs
  void apply_env_overlay(const std::string& prefix = "UAII_");

  [[nodiscard]] bool has(const std::string& key) const;

  [[nodiscard]] std::string get_string(const std::string& key,
                                       const std::string& default_value = {}) const;

  [[nodiscard]] bool get_bool(const std::string& key, bool default_value = false) const;

  [[nodiscard]] int get_int(const std::string& key, int default_value = 0) const;

  void set(const std::string& key, std::string value);

  [[nodiscard]] const std::unordered_map<std::string, std::string>& entries() const noexcept {
    return entries_;
  }

  /// Split a comma-separated config value into trimmed tokens.
  [[nodiscard]] static std::vector<std::string> split_list(const std::string& value);

 private:
  std::unordered_map<std::string, std::string> entries_;
};

/// Resolve default config search paths (cwd, ./configs, $UAII_HOME, etc.).
[[nodiscard]] UAII_API std::vector<std::string> default_config_search_paths();

/// Load first existing config from search paths, then apply env overlay.
[[nodiscard]] UAII_API Error load_default_config(Config* out);

}  // namespace uaii
