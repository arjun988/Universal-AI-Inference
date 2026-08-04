#include "commands/ir_commands.hpp"

#include "uaii/core/log.hpp"
#include "uaii/ir/ir.hpp"

#include <iostream>

namespace uaii {
namespace cli {
namespace {

void print_validate_usage() {
  std::cout
      << "Usage: uaii validate <path> [--allow-unknown-ops] [--json]\n"
      << "  Validate a UAII IR graph (JSON or binary .uaii).\n";
}

void print_inspect_usage() {
  std::cout
      << "Usage: uaii inspect <path> [--summary]\n"
      << "  Inspect tensors, nodes, and metadata of a UAII IR graph.\n";
}

void print_graph_usage() {
  std::cout
      << "Usage: uaii graph <path> [--format text|dot|json|plan]\n"
      << "  Dump the IR graph or its derived execution plan.\n";
}

std::string take_path(const std::vector<std::string>& args, std::size_t* index) {
  if (*index >= args.size()) {
    return {};
  }
  return args[(*index)++];
}

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
  for (const auto& a : args) {
    if (a == flag) {
      return true;
    }
  }
  return false;
}

std::string get_option(const std::vector<std::string>& args,
                       const std::string& key,
                       const std::string& default_value) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      return args[i + 1];
    }
  }
  return default_value;
}

}  // namespace

int cmd_validate(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    print_validate_usage();
    return args.empty() ? 1 : 0;
  }

  std::size_t i = 0;
  const std::string path = take_path(args, &i);
  if (path.empty() || path[0] == '-') {
    print_validate_usage();
    return 1;
  }

  ir::Graph graph;
  Error err = ir::load_graph(path, &graph);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  ir::ValidationOptions options;
  options.allow_unknown_ops = has_flag(args, "--allow-unknown-ops");
  const ir::ValidationResult result =
      ir::validate_graph(graph, ir::default_registry(), options);

  if (has_flag(args, "--json")) {
    std::cout << "{\n  \"ok\": " << (result.ok() ? "true" : "false")
              << ",\n  \"errors\": " << result.error_count()
              << ",\n  \"issues\": [\n";
    for (std::size_t n = 0; n < result.issues.size(); ++n) {
      const auto& issue = result.issues[n];
      std::cout << "    {\"severity\": \""
                << (issue.severity == ir::ValidationSeverity::Error ? "error" : "warning")
                << "\", \"code\": \"" << issue.code << "\", \"message\": \""
                << issue.message << "\"}";
      if (n + 1 < result.issues.size()) {
        std::cout << ",";
      }
      std::cout << "\n";
    }
    std::cout << "  ]\n}\n";
  } else {
    std::cout << ir::graph_summary(graph);
    if (result.issues.empty()) {
      std::cout << "Validation: OK\n";
    } else {
      for (const auto& issue : result.issues) {
        const char* sev =
            issue.severity == ir::ValidationSeverity::Error ? "error" : "warn";
        std::cout << "  [" << sev << "] " << issue.code << ": " << issue.message
                  << '\n';
      }
      std::cout << "Validation: " << (result.ok() ? "OK (warnings only)" : "FAILED")
                << '\n';
    }
  }

  // Also ensure an execution plan can be built when valid.
  if (result.ok()) {
    ir::ExecutionPlan plan;
    err = ir::build_execution_plan(graph, &plan);
    if (!err.ok()) {
      std::cerr << "plan: " << err.to_string() << '\n';
      return 1;
    }
    log::info("validate") << "execution plan ops=" << plan.ops.size();
  }

  return result.ok() ? 0 : 2;
}

int cmd_inspect(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    print_inspect_usage();
    return args.empty() ? 1 : 0;
  }

  std::size_t i = 0;
  const std::string path = take_path(args, &i);
  if (path.empty() || path[0] == '-') {
    print_inspect_usage();
    return 1;
  }

  ir::Graph graph;
  Error err = ir::load_graph(path, &graph);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  if (has_flag(args, "--summary")) {
    std::cout << ir::graph_summary(graph);
  } else {
    std::cout << ir::graph_to_text(graph);
  }

  // Show registered builtin ops count for context.
  std::cout << "\nOperator registry schemas: "
            << ir::default_registry().schema_count() << '\n';
  return 0;
}

int cmd_graph(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    print_graph_usage();
    return args.empty() ? 1 : 0;
  }

  std::size_t i = 0;
  const std::string path = take_path(args, &i);
  if (path.empty() || path[0] == '-') {
    print_graph_usage();
    return 1;
  }

  const std::string format = get_option(args, "--format", "text");

  ir::Graph graph;
  Error err = ir::load_graph(path, &graph);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  if (format == "text") {
    std::cout << ir::graph_to_text(graph);
  } else if (format == "dot") {
    std::cout << ir::graph_to_dot(graph);
  } else if (format == "json") {
    std::string json;
    err = ir::graph_to_json(graph, &json);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
    std::cout << json;
  } else if (format == "plan") {
    err = ir::validate_graph_error(graph, ir::default_registry());
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 2;
    }
    ir::ExecutionPlan plan;
    err = ir::build_execution_plan(graph, &plan);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
    std::cout << ir::plan_to_text(plan);
    std::cout << "\n" << ir::plan_to_json(plan);
  } else {
    std::cerr << "Unknown --format '" << format
              << "' (expected text|dot|json|plan)\n";
    return 1;
  }
  return 0;
}

}  // namespace cli
}  // namespace uaii
