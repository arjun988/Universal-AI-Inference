#include "uaii/core/config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace uaii {
namespace {

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string strip_inline_comment(const std::string& line) {
  bool in_string = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      in_string = !in_string;
      continue;
    }
    if (!in_string && c == '#') {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string unquote(std::string value) {
  value = trim(std::move(value));
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::string to_lower(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string env_key_to_config_key(std::string env_key, const std::string& prefix) {
  if (env_key.rfind(prefix, 0) != 0) {
    return {};
  }
  std::string rest = env_key.substr(prefix.size());
  // UAII_PLUGIN__DIRS -> plugin.dirs ; UAII_LOG_LEVEL -> log.level
  std::string out;
  out.reserve(rest.size());
  for (std::size_t i = 0; i < rest.size(); ++i) {
    if (rest[i] == '_' && i + 1 < rest.size() && rest[i + 1] == '_') {
      out.push_back('.');
      ++i;
      continue;
    }
    if (rest[i] == '_') {
      out.push_back('.');
    } else {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(rest[i]))));
    }
  }
  return out;
}

}  // namespace

Error Config::load_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return Error::make(ErrorCode::NotFound, "config file not found: " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return parse(ss.str(), path);
}

Error Config::parse(const std::string& text, const std::string& source_name) {
  std::string section;
  std::istringstream in(text);
  std::string raw_line;
  int line_no = 0;

  while (std::getline(in, raw_line)) {
    ++line_no;
    std::string line = trim(strip_inline_comment(raw_line));
    if (line.empty()) {
      continue;
    }

    if (line.front() == '[' && line.back() == ']') {
      section = trim(line.substr(1, line.size() - 2));
      if (section.empty()) {
        return Error::make(ErrorCode::ConfigError,
                           source_name + ":" + std::to_string(line_no) +
                               ": empty section name");
      }
      continue;
    }

    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      return Error::make(ErrorCode::ConfigError,
                         source_name + ":" + std::to_string(line_no) +
                             ": expected key = value");
    }

    std::string key = trim(line.substr(0, eq));
    std::string value = unquote(line.substr(eq + 1));
    if (key.empty()) {
      return Error::make(ErrorCode::ConfigError,
                         source_name + ":" + std::to_string(line_no) +
                             ": empty key");
    }

    const std::string full_key = section.empty() ? key : (section + "." + key);
    entries_[full_key] = std::move(value);
  }

  return Error::success();
}

void Config::apply_env_overlay(const std::string& prefix) {
#if defined(_WIN32)
  // _wenviron / GetEnvironmentStrings is heavier; walk common UAII_* via known
  // pattern is not possible without enumeration. Use GetEnvironmentStringsA.
  LPCH env_block = GetEnvironmentStringsA();
  if (env_block == nullptr) {
    return;
  }
  for (LPCH p = env_block; *p != '\0'; ) {
    std::string entry(p);
    p += entry.size() + 1;
    const auto pos = entry.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = entry.substr(0, pos);
    std::string value = entry.substr(pos + 1);
    std::string cfg_key = env_key_to_config_key(key, prefix);
    if (!cfg_key.empty()) {
      entries_[cfg_key] = std::move(value);
    }
  }
  FreeEnvironmentStringsA(env_block);
#else
  extern char** environ;
  if (environ == nullptr) {
    return;
  }
  for (char** ep = environ; *ep != nullptr; ++ep) {
    std::string entry(*ep);
    const auto pos = entry.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = entry.substr(0, pos);
    std::string value = entry.substr(pos + 1);
    std::string cfg_key = env_key_to_config_key(key, prefix);
    if (!cfg_key.empty()) {
      entries_[cfg_key] = std::move(value);
    }
  }
#endif
}

bool Config::has(const std::string& key) const {
  return entries_.find(key) != entries_.end();
}

std::string Config::get_string(const std::string& key,
                               const std::string& default_value) const {
  const auto it = entries_.find(key);
  if (it == entries_.end()) {
    return default_value;
  }
  return it->second;
}

bool Config::get_bool(const std::string& key, bool default_value) const {
  const auto it = entries_.find(key);
  if (it == entries_.end()) {
    return default_value;
  }
  const std::string v = to_lower(it->second);
  if (v == "1" || v == "true" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "0" || v == "false" || v == "no" || v == "off") {
    return false;
  }
  return default_value;
}

int Config::get_int(const std::string& key, int default_value) const {
  const auto it = entries_.find(key);
  if (it == entries_.end()) {
    return default_value;
  }
  try {
    return std::stoi(it->second);
  } catch (...) {
    return default_value;
  }
}

void Config::set(const std::string& key, std::string value) {
  entries_[key] = std::move(value);
}

std::vector<std::string> Config::split_list(const std::string& value) {
  std::vector<std::string> out;
  std::string current;
  for (char c : value) {
    if (c == ',' || c == ';') {
      current = trim(current);
      if (!current.empty()) {
        out.push_back(current);
      }
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  current = trim(current);
  if (!current.empty()) {
    out.push_back(current);
  }
  return out;
}

std::vector<std::string> default_config_search_paths() {
  std::vector<std::string> paths = {
      "uaii.toml",
      "configs/uaii.toml",
  };
  if (const char* home = std::getenv("UAII_HOME")) {
    paths.push_back(std::string(home) + "/uaii.toml");
  }
  return paths;
}

Error load_default_config(Config* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "config out pointer is null");
  }

  Error last = Error::make(ErrorCode::NotFound, "no config file found");
  for (const auto& path : default_config_search_paths()) {
    Error err = out->load_file(path);
    if (err.ok()) {
      out->apply_env_overlay();
      return Error::success();
    }
    if (err.code() != ErrorCode::NotFound) {
      return err;
    }
    last = err;
  }

  // No file is acceptable; env-only config still applies.
  out->apply_env_overlay();
  (void)last;
  return Error::success();
}

}  // namespace uaii
