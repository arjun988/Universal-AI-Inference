#include "commands/doctor.hpp"

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
      << "  help       Show this help\n"
      << "  version    Print version\n\n"
      << "Global options:\n"
      << "  --config <path>     Load config file\n"
      << "  --log-level <lvl>   trace|debug|info|warn|error|off\n"
      << "  --no-color          Disable ANSI colors\n\n"
      << "doctor options:\n"
      << "  --load-plugins      Load discovered plugins (default: discover only)\n";
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
      opts.command = "help";
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

  std::cerr << "Unknown command: " << command << "\n\n";
  print_usage(argv[0]);
  return 1;
}
