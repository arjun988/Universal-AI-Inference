#include "uaii/tokenizers/gguf_tokenizer.hpp"

#include "uaii/loaders/gguf.hpp"
#include "uaii/tokenizers/bpe_tokenizer.hpp"
#include "uaii/tokenizers/simple_tokenizer.hpp"

namespace uaii {
namespace tokenizers {
namespace {

std::vector<std::string> kv_string_array(const loaders::GgufFile& file, const char* key) {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return {};
  const auto& v = it->second;
  if (std::holds_alternative<std::vector<std::string>>(v)) {
    return std::get<std::vector<std::string>>(v);
  }
  return {};
}

}  // namespace

std::unique_ptr<ITokenizer> load_gguf_tokenizer(const std::string& path, Error* err_out) {
  loaders::GgufFile file;
  Error err = loaders::gguf_read_header(path, &file);
  if (!err.ok()) {
    if (err_out) *err_out = err;
    return nullptr;
  }

  const auto tokens = kv_string_array(file, "tokenizer.ggml.tokens");
  if (tokens.empty()) {
    err = Error::make(ErrorCode::NotFound,
                      "GGUF has no tokenizer.ggml.tokens: " + path);
    if (err_out) *err_out = err;
    return nullptr;
  }

  const auto merges = kv_string_array(file, "tokenizer.ggml.merges");
  if (!merges.empty()) {
    auto tok = std::make_unique<BpeTokenizer>();
    err = tok->load_from_vocab_merges(tokens, merges);
    if (!err.ok()) {
      if (err_out) *err_out = err;
      return nullptr;
    }
    if (err_out) *err_out = Error::success();
    return tok;
  }

  auto tok = std::make_unique<SimpleTokenizer>();
  err = tok->load_from_token_list(tokens);
  if (!err.ok()) {
    if (err_out) *err_out = err;
    return nullptr;
  }
  if (err_out) *err_out = Error::success();
  return tok;
}

}  // namespace tokenizers
}  // namespace uaii
