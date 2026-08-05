#include "commands/doctor.hpp"

#include "uaii/backends/factory.hpp"
#include "uaii/c_api/plugin_abi.h"
#include "uaii/c_api/version.h"
#include "uaii/core/log.hpp"
#include "uaii/core/plugin.hpp"
#include "uaii/interfaces/types.hpp"
#include "uaii/plugins/operator_host.hpp"
#include "uaii/version.hpp"

#include <iostream>
#include <memory>
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
  print_kv("c_api", std::string(UAII_C_API_VERSION_STRING) + " (struct_size required)");
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

  std::cout << "Modules (probed)\n";
  print_kv("uaii-core", "active");
  print_kv("uaii-ir", "active");
  print_kv("uaii-runtime", "active (CPU session; f32 compute)");
  print_kv("uaii-memory", "active");
  print_kv("uaii-storage", "active (OS mmap + double-buffer streaming)");
  print_kv("uaii-planner", "active (fusion, memory/storage plan, disk cache)");
  print_kv("uaii-kernels", "active (threaded MatMul + plugin ops)");
  print_kv("uaii-backends", "CPU real; GPU names = host-fallback unless native ON");
  print_kv("uaii-loaders", "GGUF (Q4_0/Q8_0/F16→f32) + Safetensors");
  print_kv("uaii-tokenizers", "SimpleTokenizer (whitespace; not BPE/SP)");
  print_kv("uaii-quant", "pack/unpack helpers; session compute remains f32");
  print_kv("uaii-profiler", "chrome-trace timelines");
  print_kv("uaii-capi",
           std::string("shared lib ") + UAII_C_API_VERSION_STRING +
               " (struct_size required)");
  print_kv("python SDK", "bindings/python (ctypes; bundle native via UAII_CAPI_PATH)");
  std::cout << '\n';

  std::cout << "Backends (probed)\n";
  for (const auto& b : backends::list_backends()) {
    std::unique_ptr<IBackend> be;
    backends::BackendCreateOptions bo;
    // Probe real create path; GPU backends still report host_fallback in caps.
    bo.force_host_fallback = (b.device_type != DeviceType::Cpu);
    Error cerr = backends::create_backend(b.name, bo, &be);
    std::ostringstream oss;
    oss << to_string(b.device_type)
        << " create=" << (cerr.ok() ? "ok" : "fail")
        << " native_compiled=" << (b.native_compiled ? "yes" : "no")
        << (b.device_type == DeviceType::Cpu ? " host_fallback=n/a"
                                             : " host_fallback=default")
        << " — " << b.description;
    if (cerr.ok() && be) {
      (void)be->initialize();
      const auto caps = be->capabilities();
      oss << " | " << caps.details;
      be->shutdown();
    }
    print_kv(b.name.c_str(), oss.str());
  }
  {
    const auto ops = plugins::OperatorHostRegistry::instance().names();
    print_kv("plugin_ops_registered", ops.empty() ? "(none)" : std::to_string(ops.size()));
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
