#include "commands/doctor.hpp"

#include "uaii/backends/factory.hpp"
#include "uaii/c_api/plugin_abi.h"
#include "uaii/c_api/version.h"
#include "uaii/core/log.hpp"
#include "uaii/core/plugin.hpp"
#include "uaii/version.hpp"

#include <iostream>
#include <sstream>

namespace uaii {
namespace cli {
namespace {

void print_kv(const char* key, const std::string& value) {
  std::cout << "  " << key << ": " << value << '\n';
}

std::string join_dirs(const std::vector<std::string>& dirs) {
  if (dirs.empty()) {
    return "(none)";
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < dirs.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << dirs[i];
  }
  return oss.str();
}

}  // namespace

int cmd_doctor(const Config& config, bool load_plugins, const std::string& exe_dir) {
  std::cout << "UAII Runtime doctor\n";
  std::cout << "===================\n\n";

  std::cout << "Version\n";
  print_kv("uaii", version_string());
  print_kv("c_api", UAII_C_API_VERSION_STRING);
  print_kv("plugin_abi", std::to_string(UAII_PLUGIN_ABI_VERSION));
#if defined(_WIN32)
  print_kv("platform", "windows");
#elif defined(__APPLE__)
  print_kv("platform", "macos");
#else
  print_kv("platform", "linux");
#endif
#if defined(_MSC_VER)
  print_kv("compiler", "msvc");
#elif defined(__clang__)
  print_kv("compiler", "clang");
#elif defined(__GNUC__)
  print_kv("compiler", "gcc");
#else
  print_kv("compiler", "unknown");
#endif
  std::cout << '\n';

  std::cout << "Configuration\n";
  print_kv("log.level", config.get_string("log.level", "info"));
  print_kv("log.color", config.get_bool("log.color", true) ? "true" : "false");

  auto dirs = Config::split_list(config.get_string(
      "plugin.dirs", "plugins,build/plugins/example_probe,build/lib,build/bin"));
  if (dirs.empty()) {
    dirs = {"plugins", "build/plugins/example_probe", "build/lib", "build/bin"};
  }
  if (!exe_dir.empty()) {
    dirs.insert(dirs.begin(), exe_dir + "/plugins");
    dirs.insert(dirs.begin(), exe_dir);
  }
  print_kv("plugin.dirs", join_dirs(dirs));
  std::cout << '\n';

  std::cout << "Modules (Phase 1)\n";
  print_kv("uaii-core", "active");
  print_kv("uaii-ir", "active (Phase 2)");
  print_kv("uaii-runtime", "active (Phase 3 CPU session)");
  print_kv("uaii-memory", "active (Phase 3)");
  print_kv("uaii-storage", "active (file provider + streaming)");
  print_kv("uaii-planner", "active (fusion, memory/storage plan, cache)");
  print_kv("uaii-kernels", "active (Phase 3 CPU + MatMulRelu)");
  print_kv("uaii-backends", "active (CPU + CUDA/Metal/Vulkan/WebGPU/ROCm)");
  print_kv("uaii-loaders", "active (GGUF + Safetensors)");
  print_kv("uaii-tokenizers", "active (SimpleTokenizer)");
  print_kv("uaii-quant", "active (F16/BF16/INT8/INT4/NF4/MXFP4)");
  print_kv("uaii-profiler", "active (chrome-trace timelines)");
  print_kv("uaii-capi", "active (C API 1.0.0 shared lib)");
  print_kv("python SDK", "bindings/python (ctypes + optional pybind11)");
  std::cout << '\n';

  std::cout << "Backends (Phase 5)\n";
  for (const auto& b : backends::list_backends()) {
    std::ostringstream oss;
    oss << to_string(b.device_type)
        << " available=" << (b.always_available ? "yes" : "no")
        << " native_compiled=" << (b.native_compiled ? "yes" : "no")
        << " — " << b.description;
    print_kv(b.name.c_str(), oss.str());
  }
  std::cout << '\n';

  std::cout << "Interfaces\n";
  print_kv("IBackend", "cpu/cuda/metal/vulkan/webgpu/rocm");
  print_kv("IModelLoader", "GGUF + Safetensors");
  print_kv("IOperator / IOperatorRegistry", "declared");
  print_kv("IStorageProvider", "declared");
  print_kv("IScheduler", "CPU");
  print_kv("ITokenizer", "SimpleTokenizer");
  std::cout << '\n';

  PluginRegistry registry;
  Error discover_err = registry.discover(dirs);
  if (!discover_err.ok()) {
    log::error("doctor") << discover_err.to_string();
    return 1;
  }

  std::cout << "Plugins discovered: " << registry.discovered().size() << '\n';
  for (const auto& d : registry.discovered()) {
    std::cout << "  - " << (d.name.empty() ? "(unnamed)" : d.name)
              << " kind=" << to_string(d.kind) << " abi=" << d.abi_version
              << " path=" << d.path << '\n';
    if (!d.description.empty()) {
      std::cout << "      " << d.description << '\n';
    }
  }
  std::cout << '\n';

  if (load_plugins) {
    Error load_err = registry.load_all();
    if (!load_err.ok()) {
      log::warn("doctor") << load_err.to_string();
    }
    std::cout << "Plugins loaded: " << registry.loaded_count() << '\n';
    for (const auto& p : registry.plugins()) {
      const auto& d = p->descriptor();
      std::cout << "  - " << d.name << " v" << d.version << " OK\n";
    }
  } else {
    std::cout << "Plugins loaded: (skipped; pass --load-plugins)\n";
  }

  std::cout << "\nDoctor finished.\n";
  return 0;
}

}  // namespace cli
}  // namespace uaii
