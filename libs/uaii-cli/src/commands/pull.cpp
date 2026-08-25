#include "commands/pull.hpp"

#include "uaii/core/error.hpp"
#include "uaii/core/log.hpp"
#include "uaii/loaders/hf.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace uaii {
namespace cli {
namespace {

namespace fs = std::filesystem;

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

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  uaii pull <org/model> [--outdir <dir>] [--revision <rev>]\n\n"
      << "Downloads a Hugging Face model snapshot via huggingface-cli or curl.\n"
      << "Requires network access. Prefers `huggingface-cli download` when installed.\n\n"
      << "Examples:\n"
      << "  uaii pull hf-internal-testing/tiny-random-LlamaForCausalLM\n"
      << "  uaii pull Qwen/Qwen2.5-0.5B-Instruct --outdir ./models/qwen\n";
}

int run_cmd(const std::string& cmd) {
  log::info("pull") << cmd;
  return std::system(cmd.c_str());
}

bool command_exists(const char* name) {
#if defined(_WIN32)
  const std::string probe = std::string("where ") + name + " >nul 2>nul";
#else
  const std::string probe = std::string("command -v ") + name + " >/dev/null 2>&1";
#endif
  return std::system(probe.c_str()) == 0;
}

int curl_download(const std::string& url, const std::string& dest, bool optional) {
  std::ostringstream cmd;
#if defined(_WIN32)
  cmd << "curl -fsSL -o \"" << dest << "\" \"" << url << "\"";
#else
  cmd << "curl -fsSL -o '" << dest << "' '" << url << "'";
#endif
  const int rc = run_cmd(cmd.str());
  if (rc != 0) {
    std::error_code ec2;
    fs::remove(dest, ec2);
    if (!optional) {
      log::warn("pull") << "missing: " << dest;
    } else {
      log::warn("pull") << "optional/missing: " << url;
    }
  }
  return rc;
}

bool file_nonempty(const fs::path& p) {
  std::error_code ec;
  if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) return false;
  return fs::file_size(p, ec) > 0 && !ec;
}

std::string python_bin() {
  if (command_exists("python")) return "python";
  if (command_exists("python3")) return "python3";
  return {};
}

}  // namespace

int cmd_pull(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    print_usage();
    return args.empty() ? 1 : 0;
  }

  std::string repo;
  for (const auto& a : args) {
    if (!a.empty() && a[0] != '-') {
      repo = a;
      break;
    }
  }
  if (repo.empty()) {
    std::cerr << "missing repo id (org/model)\n";
    print_usage();
    return 1;
  }

  const std::string revision = get_opt(args, "--revision", "main");
  std::string outdir = get_opt(args, "--outdir");
  if (outdir.empty()) {
    const char* home = std::getenv("UAII_MODEL_CACHE");
    if (home && home[0]) {
      outdir = (fs::path(home) / repo).string();
    } else {
      outdir = (fs::path("models") / "hf" / repo).string();
    }
  }

  std::error_code ec;
  fs::create_directories(outdir, ec);
  if (ec) {
    std::cerr << "cannot create outdir: " << outdir << "\n";
    return 1;
  }

  // Prefer huggingface-cli (handles shards + auth).
  if (command_exists("huggingface-cli")) {
    std::ostringstream cmd;
    cmd << "huggingface-cli download " << repo << " --revision " << revision
        << " --local-dir \"" << outdir << "\"";
    const int rc = run_cmd(cmd.str());
    if (rc != 0) {
      std::cerr << "huggingface-cli download failed (exit " << rc << ")\n";
      return rc == 0 ? 1 : rc;
    }
    std::cout << "pulled " << repo << " -> " << outdir << "\n";
    return 0;
  }

  if (command_exists("hf")) {
    std::ostringstream cmd;
    cmd << "hf download " << repo << " --revision " << revision << " --local-dir \""
        << outdir << "\"";
    const int rc = run_cmd(cmd.str());
    if (rc != 0) {
      std::cerr << "hf download failed (exit " << rc << ")\n";
      return rc == 0 ? 1 : rc;
    }
    std::cout << "pulled " << repo << " -> " << outdir << "\n";
    return 0;
  }

  const std::string py = python_bin();
  if (!py.empty()) {
    std::ostringstream cmd;
    cmd << py
        << " -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id=r'"
        << repo << "', revision=r'" << revision << "', local_dir=r'" << outdir << "')\"";
    const int rc = run_cmd(cmd.str());
    if (rc == 0 && fs::exists(fs::path(outdir) / "config.json")) {
      std::cout << "pulled " << repo << " -> " << outdir << "\n";
      return 0;
    }
    log::warn("pull") << "huggingface_hub snapshot unavailable; falling back to curl";
  }

  if (!command_exists("curl")) {
    std::cerr
        << "neither huggingface-cli/hf nor curl is available.\n"
        << "Install Hugging Face CLI: pip install -U huggingface_hub\n"
        << "Then: huggingface-cli download " << repo << " --local-dir " << outdir << "\n";
    return 1;
  }

  const std::string base =
      "https://huggingface.co/" + repo + "/resolve/" + revision + "/";
  const char* meta[] = {"config.json", "tokenizer.json", "tokenizer_config.json",
                        "tokenizer.model", "vocab.json", "merges.txt",
                        "model.safetensors.index.json", nullptr};
  for (int i = 0; meta[i]; ++i) {
    const bool required = std::string(meta[i]) == "config.json";
    curl_download(base + meta[i], (fs::path(outdir) / meta[i]).string(), !required);
  }

  if (!file_nonempty(fs::path(outdir) / "config.json")) {
    std::cerr << "pull incomplete: config.json missing.\n";
    return 1;
  }

  const fs::path index_path = fs::path(outdir) / "model.safetensors.index.json";
  std::vector<std::string> shards;
  if (file_nonempty(index_path)) {
    std::ifstream in(index_path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    Error err = loaders::hf_index_unique_shards(ss.str(), &shards);
    if (!err.ok()) {
      std::cerr << "pull: " << err.to_string() << "\n";
      return 1;
    }
  } else {
    shards.push_back("model.safetensors");
  }

  for (const auto& shard : shards) {
    const fs::path dest = fs::path(outdir) / shard;
    if (file_nonempty(dest)) continue;
    const int rc = curl_download(base + shard, dest.string(), false);
    if (rc != 0 || !file_nonempty(dest)) {
      std::cerr << "pull incomplete: missing shard " << shard << "\n";
      return 1;
    }
  }

  {
    Error verr = loaders::hf_verify_shards_present(outdir, shards);
    if (!verr.ok()) {
      std::cerr << verr.message() << "\n";
      return 1;
    }
  }

  std::cout << "pulled " << repo << " -> " << outdir << "\n";
  return 0;
}

}  // namespace cli
}  // namespace uaii
