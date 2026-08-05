#include "commands/convert.hpp"

#include "uaii/loaders/registry.hpp"
#include "uaii/tokenizers/bpe_tokenizer.hpp"
#include "uaii/tokenizers/gguf_tokenizer.hpp"
#include "uaii/tokenizers/sentencepiece_tokenizer.hpp"
#include "uaii/tokenizers/simple_tokenizer.hpp"

#include <iostream>
#include <memory>
#include <sstream>

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

std::string supported_convert_formats() {
  std::ostringstream oss;
  bool first = true;
  for (const auto& info : loaders::default_loaders().list()) {
    for (const auto& ext : info.supported_extensions) {
      if (!first) oss << ", ";
      first = false;
      oss << ext;
    }
  }
  oss << " (MLX: directory with config.json + .safetensors; "
         "PyTorch: .pt/.pth + sidecar .onnx or .uaii.json)";
  return oss.str();
}

bool is_value_flag(const std::string& arg) {
  return arg == "--vocab" || arg == "--bpe" || arg == "--merges" || arg == "--sp" ||
         arg == "--gguf";
}

Error make_tokenizer(const std::vector<std::string>& args,
                     std::unique_ptr<ITokenizer>* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tokenizer out null");
  }

  const std::string gguf_path = get_opt(args, "--gguf");
  if (!gguf_path.empty()) {
    Error err;
    auto tok = tokenizers::load_gguf_tokenizer(gguf_path, &err);
    if (!err.ok()) return err;
    *out = std::move(tok);
    return Error::success();
  }

  const std::string bpe_vocab = get_opt(args, "--bpe");
  const std::string merges_path = get_opt(args, "--merges");
  if (!bpe_vocab.empty() || !merges_path.empty()) {
    if (bpe_vocab.empty() || merges_path.empty()) {
      return Error::make(ErrorCode::InvalidArgument,
                         "BPE requires both --bpe <vocab.json> and --merges <merges.txt>");
    }
    auto tok = std::make_unique<tokenizers::BpeTokenizer>();
    Error err = tok->load(bpe_vocab, merges_path);
    if (!err.ok()) return err;
    *out = std::move(tok);
    return Error::success();
  }

  const std::string sp_model = get_opt(args, "--sp");
  if (!sp_model.empty()) {
    auto tok = std::make_unique<tokenizers::SentencePieceTokenizer>();
    Error err = tok->load(sp_model);
    if (!err.ok()) return err;
    *out = std::move(tok);
    return Error::success();
  }

  const std::string vocab_path = get_opt(args, "--vocab");
  auto tok = std::make_unique<tokenizers::SimpleTokenizer>(
      vocab_path.empty() ? tokenizers::SimpleTokenizer::demo_vocab()
                         : tokenizers::SimpleTokenizer{});
  if (!vocab_path.empty()) {
    Error err = tok->load_vocab_file(vocab_path);
    if (!err.ok()) return err;
  }
  *out = std::move(tok);
  return Error::success();
}

}  // namespace

int cmd_convert(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    std::cout
        << "Usage: uaii convert <input> -o <output.uaii.json|.uaii>\n"
        << "  Convert external model formats to UAII IR.\n"
        << "  Supported inputs: " << supported_convert_formats() << '\n';
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

  auto* loader = loaders::default_loaders().find_for_path(input);
  if (loader == nullptr) {
    std::cerr << "no loader accepts path: " << input << '\n'
              << "supported: " << supported_convert_formats() << '\n';
    return 1;
  }

  Error err = loaders::convert_model(input, output);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }
  std::cout << "Wrote " << output << " via " << loader->info().name << " loader\n";
  return 0;
}

int cmd_tokenize(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    std::cout
        << "Usage:\n"
        << "  uaii tokenize encode <text>\n"
        << "  uaii tokenize decode <id,id,...>\n\n"
        << "Tokenizer selection (default: SimpleTokenizer demo vocab):\n"
        << "  --vocab <file>                 line-delimited vocab (SimpleTokenizer)\n"
        << "  --bpe <vocab.json> --merges <merges.txt>   GPT-2 style BPE\n"
        << "  --gguf <model.gguf>            tokenizer.ggml.tokens[/merges] from GGUF\n"
#if defined(UAII_HAVE_SENTENCEPIECE)
        << "  --sp <model.model>             SentencePiece model\n"
#else
        << "  --sp <model.model>             SentencePiece (build with UAII_WITH_SENTENCEPIECE)\n"
#endif
        ;
    return args.empty() ? 1 : 0;
  }

  std::string mode;
  std::vector<std::string> positional;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (is_value_flag(args[i])) {
      ++i;
      continue;
    }
    if (!args[i].empty() && args[i][0] == '-') continue;
    if (mode.empty()) {
      mode = args[i];
    } else {
      positional.push_back(args[i]);
    }
  }

  std::unique_ptr<ITokenizer> tok;
  Error err = make_tokenizer(args, &tok);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
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
    err = tok->encode(text, &ids);
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
    err = tok->decode(ids, &text);
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
