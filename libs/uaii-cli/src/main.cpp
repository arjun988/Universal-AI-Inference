#include "commands/convert.hpp"
#include "commands/doctor.hpp"
#include "commands/generate.hpp"
#include "commands/ir_commands.hpp"
#include "commands/optimize.hpp"
#include "commands/run.hpp"

#include "uaii/core/config.hpp"
#include "uaii/core/log.hpp"
#include "uaii/version.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
  std::cout
      << "Universal AI Inference Runtime (UAII) " << uaii::version_string() << "\n\n"
      << "Usage:\n"
      << "  " << argv0 << " <command> [options]\n\n"
      << "Commands:\n"
      << "  doctor     Diagnose environment, modules, and plugins\n"
      << "  validate   Validate a UAII IR graph\n"
      << "  inspect    Inspect tensors / nodes / metadata\n"
      << "  graph      Dump IR graph (text|dot|json|plan)\n"
      << "  convert    GGUF/Safetensors/ONNX/MLX/PyTorch sidecar → UAII IR\n"
      << "  tokenize   Encode/decode (Simple/BPE/SentencePiece/GGUF)\n"
      << "  generate   Text generation from a GGUF model (greedy)\n"
      << "  chat       JSONL chat worker (warm session for dashboard)\n"
      << "  run        Execute IR (or built-in demos)\n"
      << "  profile    Capture chrome-trace profiler JSON\n"
      << "  benchmark  Baseline vs optimized timings\n"
      << "  cache      Plan-cache status/clear\n"
      << "  help       Show this help\n"
      << "  version    Print version\n\n"
      << "Phase 6:\n"
      << "  " << argv0 << " run --demo optimize\n"
      << "  " << argv0 << " run --demo streaming\n"
      << "  " << argv0 << " run --demo profile\n"
      << "  " << argv0 << " run --demo quant --format int8\n"
      << "  " << argv0 << " benchmark --demo\n"
      << "  " << argv0 << " profile --demo\n";
}

struct GlobalOptions {
  std::string config_path;
  std::string log_level;
  bool no_color = false;
  bool load_plugins = false;
  std::string command;
  std::vector<std::string> rest;
};

GlobalOptions parse_args(int argc, char** argv) {
  GlobalOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      opts.config_path = argv[++i];
    } else if (arg == "--log-level" && i + 1 < argc) {
      opts.log_level = argv[++i];
    } else if (arg == "--no-color") {
      opts.no_color = true;
    } else if (arg == "--load-plugins") {
      opts.load_plugins = true;
    } else if (arg == "-h" || arg == "--help") {
      if (opts.command.empty()) {
        opts.command = "help";
      } else {
        opts.rest.push_back(arg);
      }
    } else if (!arg.empty() && arg[0] == '-') {
      opts.rest.push_back(arg);
    } else if (opts.command.empty()) {
      opts.command = arg;
    } else {
      opts.rest.push_back(arg);
    }
  }
  return opts;
}

}  // namespace

int main(int argc, char** argv) {
  const GlobalOptions opts = parse_args(argc, argv);

  uaii::Config config;
  if (!opts.config_path.empty()) {
    uaii::Error err = config.load_file(opts.config_path);
    if (!err.ok()) {
      std::cerr << "Failed to load config: " << err.to_string() << '\n';
      return 1;
    }
    config.apply_env_overlay();
  } else {
    uaii::Error err = uaii::load_default_config(&config);
    if (!err.ok()) {
      std::cerr << "Failed to load config: " << err.to_string() << '\n';
      return 1;
    }
  }

  if (opts.no_color || !config.get_bool("log.color", true)) {
    uaii::log::set_use_color(false);
  }

  std::string level_text = opts.log_level.empty()
                               ? config.get_string("log.level", "info")
                               : opts.log_level;
  uaii::log::Level level = uaii::log::Level::Info;
  if (!uaii::log::parse_level(level_text, &level)) {
    std::cerr << "Invalid log level: " << level_text << '\n';
    return 1;
  }
  uaii::log::set_level(level);

  const std::string command = opts.command.empty() ? "help" : opts.command;

  if (command == "help") {
    print_usage(argv[0]);
    return 0;
  }
  if (command == "version") {
    std::cout << uaii::version_string() << '\n';
    return 0;
  }
  if (command == "doctor") {
    std::string exe_dir;
    try {
      exe_dir = std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path()).string();
    } catch (...) {
      exe_dir.clear();
    }
    return uaii::cli::cmd_doctor(config, opts.load_plugins, exe_dir);
  }
  if (command == "validate") {
    return uaii::cli::cmd_validate(opts.rest);
  }
  if (command == "inspect") {
    return uaii::cli::cmd_inspect(opts.rest);
  }
  if (command == "graph") {
    return uaii::cli::cmd_graph(opts.rest);
  }
  if (command == "convert") {
    return uaii::cli::cmd_convert(opts.rest);
  }
  if (command == "tokenize") {
    return uaii::cli::cmd_tokenize(opts.rest);
  }
  if (command == "generate") {
    return uaii::cli::cmd_generate(opts.rest);
  }
  if (command == "chat") {
    return uaii::cli::cmd_chat(opts.rest);
  }
  if (command == "run") {
    return uaii::cli::cmd_run(opts.rest);
  }
  if (command == "profile") {
    return uaii::cli::cmd_profile(opts.rest);
  }
  if (command == "benchmark") {
    return uaii::cli::cmd_benchmark(opts.rest);
  }
  if (command == "cache") {
    return uaii::cli::cmd_cache(opts.rest);
  }

  std::cerr << "Unknown command: " << command << "\n\n";
  print_usage(argv[0]);
  return 1;
}
