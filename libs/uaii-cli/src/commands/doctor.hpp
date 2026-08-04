#pragma once

#include "uaii/core/config.hpp"

#include <string>

namespace uaii {
namespace cli {

/// Print environment diagnostics. Returns process exit code.
/// exe_dir: directory containing the uaii binary (used for plugin discovery).
int cmd_doctor(const Config& config, bool load_plugins, const std::string& exe_dir);

}  // namespace cli
}  // namespace uaii
