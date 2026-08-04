#include "commands/run.hpp"

#include "uaii/backends/factory.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/serialize.hpp"
#include "uaii/runtime/session.hpp"

#include <iostream>
#include <sstream>

namespace uaii {
namespace cli {
namespace {

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  uaii run --demo toy_mlp|tiny_block|gguf|safetensors|moe|parity\n"
      << "  uaii run <ir-path> --input name=v1,v2,... [options]\n\n"
      << "Options:\n"
      << "  --backend <name>          cpu|cuda|metal|vulkan|webgpu|rocm (default: cpu)\n"
      << "  --force-host-fallback     Prefer host kernels even if native compiled\n"
      << "  --output <name>           Tensor to print (default: first graph output)\n"
      << "  --weights-dir <dir>       Directory for weight_ref files\n"
      << "  --weight-init <mode>      zeros|ones|sequence (fallback if file missing)\n"
      << "  --budget-mb <n>           Memory budget in MiB (0 = unlimited)\n"
      << "  --allow-unknown-ops       Pass-through to validator\n";
}

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
  for (const auto& a : args) {
    if (a == flag) return true;
  }
  return false;
}

std::string get_opt(const std::vector<std::string>& args, const std::string& key,
                    const std::string& def = {}) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) return args[i + 1];
  }
  return def;
}

std::vector<std::string> get_opt_all(const std::vector<std::string>& args,
                                     const std::string& key) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      out.push_back(args[i + 1]);
    }
  }
  return out;
}

Error parse_f32_list(const std::string& text, std::vector<float>* out) {
  out->clear();
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    try {
      out->push_back(std::stof(item));
    } catch (...) {
      return Error::make(ErrorCode::InvalidArgument, "bad float list: " + text);
    }
  }
  return Error::ok();
}

runtime::WeightInit parse_weight_init(const std::string& s) {
  if (s == "zeros") return runtime::WeightInit::Zeros;
  if (s == "ones") return runtime::WeightInit::Ones;
  if (s == "sequence") return runtime::WeightInit::Sequence;
  return runtime::WeightInit::None;
}

void print_vector(const std::string& name, const std::vector<float>& v) {
  std::cout << name << " = [";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << v[i];
  }
  std::cout << "]\n";
}

}  // namespace

int cmd_run(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    print_usage();
    return args.empty() ? 1 : 0;
  }

  const std::string demo = get_opt(args, "--demo");
  if (!demo.empty()) {
    if (demo == "toy_mlp") {
      std::vector<float> out;
      bool matched = false;
      Error err = runtime::run_toy_mlp_demo(&out, &matched);
      if (!err.ok()) {
        std::cerr << err.to_string() << '\n';
        return 1;
      }
      print_vector("y_prob", out);
      std::cout << "expected_match: " << (matched ? "true" : "false") << '\n';
      return matched ? 0 : 2;
    }
    if (demo == "tiny_block") {
      std::vector<float> out;
      Error err = runtime::run_tiny_block_demo(&out);
      if (!err.ok()) {
        std::cerr << err.to_string() << '\n';
        return 1;
      }
      print_vector("y", out);
      return 0;
    }
    if (demo == "gguf") {
      std::string decoded;
      bool ok = false;
      Error err = runtime::run_gguf_generate_demo(&decoded, &ok);
      if (!err.ok()) {
        std::cerr << err.to_string() << '\n';
        return 1;
      }
      std::cout << "decoded: " << decoded << "\nok: " << (ok ? "true" : "false")
                << '\n';
      return ok ? 0 : 2;
    }
    if (demo == "safetensors") {
      std::string decoded;
      bool ok = false;
      Error err = runtime::run_safetensors_generate_demo(&decoded, &ok);
      if (!err.ok()) {
        std::cerr << err.to_string() << '\n';
        return 1;
      }
      std::cout << "decoded: " << decoded << "\nok: " << (ok ? "true" : "false")
                << '\n';
      return ok ? 0 : 2;
    }
    if (demo == "moe") {
      bool ok = false;
      Error err = runtime::run_moe_smoke_demo(&ok);
      if (!err.ok()) {
        std::cerr << err.to_string() << '\n';
        return 1;
      }
      std::cout << "moe_ok: " << (ok ? "true" : "false") << '\n';
      return ok ? 0 : 2;
    }
    if (demo == "parity") {
      backends::ParityReport report;
      Error err = runtime::run_parity_demo(&report);
      std::cout << "parity " << report.backend_a << " vs " << report.backend_b
                << ": " << (report.ok ? "ok" : "fail") << '\n';
      std::cout << report.message << '\n';
      for (const auto& d : report.diffs) {
        std::cout << "  " << d.name << " max_abs=" << d.max_abs_diff
                  << " max_rel=" << d.max_rel_diff
                  << " ok=" << (d.ok ? "true" : "false") << '\n';
      }
      if (!err.ok()) {
        std::cerr << err.to_string() << '\n';
        return 1;
      }
      return report.ok ? 0 : 2;
    }
    std::cerr << "Unknown demo '" << demo
              << "' (toy_mlp|tiny_block|gguf|safetensors|moe|parity)\n";
    return 1;
  }

  // Positional IR path: first non-flag arg
  std::string ir_path;
  for (const auto& a : args) {
    if (!a.empty() && a[0] != '-') {
      ir_path = a;
      break;
    }
  }
  if (ir_path.empty()) {
    print_usage();
    return 1;
  }

  ir::Graph graph;
  Error err = ir::load_graph(ir_path, &graph);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  runtime::SessionOptions opts;
  opts.weights_dir = get_opt(args, "--weights-dir");
  opts.weight_init = parse_weight_init(get_opt(args, "--weight-init", "none"));
  opts.allow_unknown_ops = has_flag(args, "--allow-unknown-ops");
  opts.backend_name = get_opt(args, "--backend", "cpu");
  opts.force_host_fallback = has_flag(args, "--force-host-fallback");
  opts.prefer_native = !opts.force_host_fallback;
  if (!backends::backend_exists(opts.backend_name)) {
    std::cerr << "unknown --backend '" << opts.backend_name << "'\n";
    return 1;
  }
  const std::string budget = get_opt(args, "--budget-mb", "0");
  try {
    const int mb = std::stoi(budget);
    if (mb > 0) {
      opts.allocator.budget_bytes =
          static_cast<std::uint64_t>(mb) * 1024ull * 1024ull;
    }
  } catch (...) {
    std::cerr << "bad --budget-mb\n";
    return 1;
  }

  runtime::Session session;
  err = session.create(std::move(graph), opts);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  for (const auto& spec : get_opt_all(args, "--input")) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos) {
      std::cerr << "--input expects name=v1,v2,...\n";
      return 1;
    }
    const std::string name = spec.substr(0, eq);
    std::vector<float> values;
    err = parse_f32_list(spec.substr(eq + 1), &values);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
    err = session.set_tensor_f32(name, values);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
  }

  err = session.run();
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  std::string out_name = get_opt(args, "--output");
  if (out_name.empty()) {
    if (session.graph().outputs.empty()) {
      std::cerr << "graph has no outputs\n";
      return 1;
    }
    const TensorId oid = session.graph().outputs.front();
    const auto* t = session.graph().find_tensor(oid);
    out_name = (t && !t->name.empty()) ? t->name : ("#" + std::to_string(oid));
  }

  std::vector<float> out;
  err = session.get_tensor_f32(out_name, &out);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }
  print_vector(out_name, out);
  log::info("run") << session.debug_stats();
  return 0;
}

}  // namespace cli
}  // namespace uaii
