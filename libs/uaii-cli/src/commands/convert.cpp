#include "commands/convert.hpp"

#include "uaii/loaders/registry.hpp"
#include "uaii/tokenizers/simple_tokenizer.hpp"

#include <iostream>

namespace uaii {
namespace cli {
namespace {

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

}  // namespace

int cmd_convert(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    std::cout
        << "Usage: uaii convert <input> -o <output.uaii.json|.uaii>\n"
        << "  Convert GGUF / Safetensors → UAII IR\n";
    return args.empty() ? 1 : 0;
  }

  std::string input;
  for (const auto& a : args) {
    if (!a.empty() && a[0] != '-') {
      input = a;
      break;
    }
  }
  const std::string output = get_opt(args, "-o", get_opt(args, "--output"));
  if (input.empty() || output.empty()) {
    std::cerr << "convert requires <input> and -o <output>\n";
    return 1;
  }

  Error err = loaders::convert_model(input, output);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }
  std::cout << "Wrote " << output << '\n';
  return 0;
}

int cmd_tokenize(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    std::cout
        << "Usage:\n"
        << "  uaii tokenize encode <text>\n"
        << "  uaii tokenize decode <id,id,...>\n"
        << "  uaii tokenize encode --vocab <file> <text>\n";
    return args.empty() ? 1 : 0;
  }

  std::string mode;
  std::string vocab_path = get_opt(args, "--vocab");
  std::vector<std::string> positional;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--vocab") {
      ++i;
      continue;
    }
    if (!args[i].empty() && args[i][0] == '-') continue;
    if (mode.empty()) mode = args[i];
    else positional.push_back(args[i]);
  }

  tokenizers::SimpleTokenizer tok =
      vocab_path.empty() ? tokenizers::SimpleTokenizer::demo_vocab()
                         : tokenizers::SimpleTokenizer{};
  if (!vocab_path.empty()) {
    Error err = tok.load_vocab_file(vocab_path);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
  }

  if (mode == "encode") {
    if (positional.empty()) {
      std::cerr << "encode requires text\n";
      return 1;
    }
    std::string text = positional[0];
    for (std::size_t i = 1; i < positional.size(); ++i) {
      text += " " + positional[i];
    }
    std::vector<std::int64_t> ids;
    Error err = tok.encode(text, &ids);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << ids[i];
    }
    std::cout << '\n';
    return 0;
  }

  if (mode == "decode") {
    if (positional.empty()) {
      std::cerr << "decode requires id list\n";
      return 1;
    }
    std::vector<std::int64_t> ids;
    std::string list = positional[0];
    std::size_t start = 0;
    while (start < list.size()) {
      auto comma = list.find(',', start);
      auto part = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
      ids.push_back(std::stoll(part));
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    std::string text;
    Error err = tok.decode(ids, &text);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
    std::cout << text << '\n';
    return 0;
  }

  std::cerr << "unknown tokenize mode (encode|decode)\n";
  return 1;
}

}  // namespace cli
}  // namespace uaii
