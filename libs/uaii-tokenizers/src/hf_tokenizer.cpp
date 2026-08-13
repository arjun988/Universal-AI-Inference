#include "uaii/tokenizers/hf_tokenizer.hpp"

#include "uaii/ir/json.hpp"
#include "uaii/tokenizers/bpe_tokenizer.hpp"
#include "uaii/tokenizers/sentencepiece_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include <filesystem>

namespace uaii {
namespace tokenizers {
namespace {

namespace fs = std::filesystem;

std::string read_file(const std::string& path, Error* err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (err) *err = Error::make(ErrorCode::NotFound, "cannot read " + path);
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (err) *err = Error::success();
  return ss.str();
}

std::string join_dir(const std::string& dir, const std::string& file) {
  return (fs::path(dir) / file).string();
}

}  // namespace

std::string load_hf_chat_template(const std::string& model_dir) {
  Error err;
  const std::string path = join_dir(model_dir, "tokenizer_config.json");
  const std::string text = read_file(path, &err);
  if (!err.ok()) return {};
  ir::json::Value root;
  if (!ir::json::parse(text, &root).ok() || !root.is_object()) return {};
  const auto* ct = ir::json::get(root.obj, "chat_template");
  if (ct && ct->is_string()) return ct->str;
  return {};
}

std::string detect_hf_chat_template_name(const std::string& jinja_or_empty,
                                         const std::string& architecture) {
  if (jinja_or_empty.find("<|start_header_id|>") != std::string::npos) return "llama3";
  if (jinja_or_empty.find("<start_of_turn>") != std::string::npos) return "gemma";
  if (jinja_or_empty.find("[INST]") != std::string::npos) return "mistral";
  if (jinja_or_empty.find("<|im_start|>") != std::string::npos) return "chatml";
  std::string a = architecture;
  for (char& c : a) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (a.find("gemma") != std::string::npos) return "gemma";
  if (a.find("mistral") != std::string::npos) return "mistral";
  if (a.find("llama") != std::string::npos) return "llama3";
  return "chatml";
}

std::unique_ptr<ITokenizer> load_hf_tokenizer(const std::string& model_dir, Error* err) {
  std::error_code ec;
  const auto vocab = join_dir(model_dir, "vocab.json");
  const auto merges = join_dir(model_dir, "merges.txt");
  if (fs::exists(vocab, ec) && fs::exists(merges, ec)) {
    auto tok = std::make_unique<BpeTokenizer>();
    Error e = tok->load(vocab, merges);
    if (!e.ok()) {
      if (err) *err = e;
      return nullptr;
    }
    if (err) *err = Error::success();
    return tok;
  }

  const auto spm = join_dir(model_dir, "tokenizer.model");
  if (fs::exists(spm, ec)) {
#if defined(UAII_HAVE_SENTENCEPIECE)
    auto tok = std::make_unique<SentencePieceTokenizer>();
    Error e = tok->load(spm);
    if (!e.ok()) {
      if (err) *err = e;
      return nullptr;
    }
    if (err) *err = Error::success();
    return tok;
#else
    if (err) {
      *err = Error::make(ErrorCode::NotImplemented,
                         "tokenizer.model found but UAII was built without SentencePiece. "
                         "Rebuild with -DUAII_WITH_SENTENCEPIECE=ON, or provide "
                         "tokenizer.json / vocab.json+merges.txt");
    }
    return nullptr;
#endif
  }

  const auto tj = join_dir(model_dir, "tokenizer.json");
  if (fs::exists(tj, ec)) {
    Error e;
    const std::string text = read_file(tj, &e);
    if (!e.ok()) {
      if (err) *err = e;
      return nullptr;
    }
    ir::json::Value root;
    e = ir::json::parse(text, &root);
    if (!e.ok() || !root.is_object()) {
      if (err) *err = Error::make(ErrorCode::InvalidArgument, "bad tokenizer.json");
      return nullptr;
    }
    const auto* model = ir::json::get(root.obj, "model");
    if (!model || !model->is_object()) {
      if (err) {
        *err = Error::make(ErrorCode::InvalidArgument, "tokenizer.json missing model object");
      }
      return nullptr;
    }
    std::vector<std::string> tokens;
    std::vector<std::string> merge_lines;
    const auto* vocab_v = ir::json::get(model->obj, "vocab");
    if (vocab_v && vocab_v->is_object()) {
      std::int64_t max_id = -1;
      for (const auto& kv : vocab_v->obj) {
        if (!kv.second.is_number()) continue;
        max_id = std::max(max_id, static_cast<std::int64_t>(kv.second.number));
      }
      if (max_id < 0) {
        if (err) *err = Error::make(ErrorCode::InvalidArgument, "empty tokenizer.json vocab");
        return nullptr;
      }
      tokens.assign(static_cast<std::size_t>(max_id + 1), std::string{});
      for (const auto& kv : vocab_v->obj) {
        if (!kv.second.is_number()) continue;
        const auto id = static_cast<std::size_t>(kv.second.number);
        if (id < tokens.size()) tokens[id] = kv.first;
      }
    }
    const auto* merges_v = ir::json::get(model->obj, "merges");
    if (merges_v && merges_v->is_array()) {
      for (const auto& m : merges_v->arr) {
        if (m.is_string()) {
          merge_lines.push_back(m.str);
        } else if (m.is_array() && m.arr.size() >= 2 && m.arr[0].is_string() &&
                   m.arr[1].is_string()) {
          merge_lines.push_back(m.arr[0].str + " " + m.arr[1].str);
        }
      }
    }
    if (tokens.empty()) {
      if (err) {
        *err = Error::make(ErrorCode::InvalidArgument,
                           "tokenizer.json has no extractable BPE vocab");
      }
      return nullptr;
    }
    auto tok = std::make_unique<BpeTokenizer>();
    e = tok->load_from_vocab_merges(tokens, merge_lines);
    if (!e.ok()) {
      if (err) *err = e;
      return nullptr;
    }
    if (err) *err = Error::success();
    return tok;
  }

  if (err) {
    *err = Error::make(ErrorCode::NotFound,
                       "no HF tokenizer found in " + model_dir +
                           " (need tokenizer.json, vocab.json+merges.txt, or tokenizer.model)");
  }
  return nullptr;
}

}  // namespace tokenizers
}  // namespace uaii
