#include "uaii/tokenizers/gguf_tokenizer.hpp"

#include "uaii/loaders/gguf.hpp"
#include "uaii/tokenizers/bpe_tokenizer.hpp"
#include "uaii/tokenizers/spm_tokenizer.hpp"

#if defined(UAII_HAVE_SENTENCEPIECE)
#  include "uaii/tokenizers/sentencepiece_tokenizer.hpp"
#endif

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

std::string kv_string(const loaders::GgufFile& file, const char* key) {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return {};
  if (std::holds_alternative<std::string>(it->second)) return std::get<std::string>(it->second);
  return {};
}

std::int64_t kv_i64(const loaders::GgufFile& file, const char* key, std::int64_t def) {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return def;
  const auto& v = it->second;
  if (std::holds_alternative<std::int64_t>(v)) return std::get<std::int64_t>(v);
  if (std::holds_alternative<std::uint64_t>(v))
    return static_cast<std::int64_t>(std::get<std::uint64_t>(v));
  if (std::holds_alternative<std::int32_t>(v)) return std::get<std::int32_t>(v);
  if (std::holds_alternative<std::uint32_t>(v))
    return static_cast<std::int64_t>(std::get<std::uint32_t>(v));
  return def;
}

bool kv_bool(const loaders::GgufFile& file, const char* key, bool def) {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return def;
  if (std::holds_alternative<bool>(it->second)) return std::get<bool>(it->second);
  return def;
}

std::string lower_ascii(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
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

  const std::string model_type = lower_ascii(kv_string(file, "tokenizer.ggml.model"));
  if (model_type == "gpt2") {
    err = Error::make(ErrorCode::NotFound,
                      "GGUF BPE model missing tokenizer.ggml.merges: " + path);
    if (err_out) *err_out = err;
    return nullptr;
  }

#if defined(UAII_HAVE_SENTENCEPIECE)
  // Optional external .model sidecar next to GGUF (user-configured export).
  if (model_type == "sentencepiece" || model_type == "spm") {
    const std::string sidecar = path.substr(0, path.find_last_of('.')) + ".model";
    auto sp = std::make_unique<SentencePieceTokenizer>();
    err = sp->load(sidecar);
    if (err.ok()) {
      if (err_out) *err_out = Error::success();
      return sp;
    }
  }
#endif

  const std::int64_t bos_id = kv_i64(file, "tokenizer.ggml.bos_token_id", -1);
  const std::int64_t eos_id = kv_i64(file, "tokenizer.ggml.eos_token_id", -1);
  const std::int64_t unk_id = kv_i64(file, "tokenizer.ggml.unknown_token_id", 0);
  const bool add_bos = kv_bool(file, "tokenizer.ggml.add_bos_token", false);
  const bool add_eos = kv_bool(file, "tokenizer.ggml.add_eos_token", false);

  auto tok = std::make_unique<SpmTokenizer>();
  err = tok->load_from_gguf_tokens(tokens, bos_id, eos_id, unk_id, add_bos, add_eos);
  if (!err.ok()) {
    if (err_out) *err_out = err;
    return nullptr;
  }
  if (err_out) *err_out = Error::success();
  return tok;
}

}  // namespace tokenizers
}  // namespace uaii
