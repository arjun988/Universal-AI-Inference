#include "uaii/tokenizers/bpe_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace uaii {
namespace tokenizers {
namespace {

// Minimal JSON object string→int parser for vocab.json {"token": id, ...}
Error parse_vocab_json(const std::string& text,
                       std::unordered_map<std::string, std::int64_t>* out) {
  out->clear();
  std::size_t i = 0;
  while (i < text.size() && text[i] != '{') ++i;
  if (i >= text.size()) return Error::make(ErrorCode::InvalidArgument, "vocab json");
  ++i;
  while (i < text.size()) {
    while (i < text.size() && (std::isspace(static_cast<unsigned char>(text[i])) || text[i] == ','))
      ++i;
    if (i < text.size() && text[i] == '}') break;
    if (i >= text.size() || text[i] != '"') {
      return Error::make(ErrorCode::InvalidArgument, "vocab key");
    }
    ++i;
    std::string key;
    while (i < text.size() && text[i] != '"') {
      if (text[i] == '\\' && i + 1 < text.size()) {
        ++i;
        key.push_back(text[i++]);
      } else {
        key.push_back(text[i++]);
      }
    }
    if (i >= text.size()) return Error::make(ErrorCode::InvalidArgument, "vocab key end");
    ++i;
    while (i < text.size() && text[i] != ':') ++i;
    if (i >= text.size()) return Error::make(ErrorCode::InvalidArgument, "vocab colon");
    ++i;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    std::size_t j = i;
    while (j < text.size() && (std::isdigit(static_cast<unsigned char>(text[j])) || text[j] == '-'))
      ++j;
    (*out)[key] = std::stoll(text.substr(i, j - i));
    i = j;
  }
  return Error::success();
}

}  // namespace

Error BpeTokenizer::load(const std::string& vocab_json_path, const std::string& merges_path) {
  std::ifstream vin(vocab_json_path);
  if (!vin) return Error::make(ErrorCode::NotFound, "vocab not found: " + vocab_json_path);
  std::ostringstream vs;
  vs << vin.rdbuf();
  Error err = parse_vocab_json(vs.str(), &token_to_id_);
  if (!err.ok()) return err;
  std::int64_t max_id = -1;
  for (const auto& kv : token_to_id_) max_id = std::max(max_id, kv.second);
  id_to_token_.assign(static_cast<std::size_t>(max_id + 1), "");
  for (const auto& kv : token_to_id_) {
    if (kv.second >= 0) id_to_token_[static_cast<std::size_t>(kv.second)] = kv.first;
  }

  std::ifstream min(merges_path);
  if (!min) return Error::make(ErrorCode::NotFound, "merges not found: " + merges_path);
  merges_.clear();
  std::string line;
  bool first = true;
  while (std::getline(min, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (first && line.rfind("#version", 0) == 0) {
      first = false;
      continue;
    }
    first = false;
    const auto sp = line.find(' ');
    if (sp == std::string::npos) continue;
    merges_.emplace_back(line.substr(0, sp), line.substr(sp + 1));
  }
  return Error::success();
}

Error BpeTokenizer::load_from_vocab_merges(const std::vector<std::string>& tokens,
                                           const std::vector<std::string>& merge_lines) {
  if (tokens.empty()) {
    return Error::make(ErrorCode::InvalidArgument, "empty gguf/bpe token list");
  }
  token_to_id_.clear();
  token_to_id_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    token_to_id_[tokens[i]] = static_cast<std::int64_t>(i);
  }
  id_to_token_ = tokens;
  merges_.clear();
  merges_.reserve(merge_lines.size());
  for (const auto& line : merge_lines) {
    const auto sp = line.find(' ');
    if (sp == std::string::npos) continue;
    merges_.emplace_back(line.substr(0, sp), line.substr(sp + 1));
  }
  return Error::success();
}

Error BpeTokenizer::encode(const std::string& text,
                           std::vector<std::int64_t>* out_tokens) const {
  if (out_tokens == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tokens out null");
  }
  out_tokens->clear();
  // Byte-level fallback: map each UTF-8 byte char as token if present, else unk=0
  std::vector<std::string> word;
  for (unsigned char c : text) {
    word.push_back(std::string(1, static_cast<char>(c)));
  }
  for (const auto& m : merges_) {
    for (;;) {
      bool merged = false;
      for (std::size_t i = 0; i + 1 < word.size(); ++i) {
        if (word[i] == m.first && word[i + 1] == m.second) {
          word[i] = m.first + m.second;
          word.erase(word.begin() + static_cast<std::ptrdiff_t>(i) + 1);
          merged = true;
          break;
        }
      }
      if (!merged) break;
    }
  }
  for (const auto& w : word) {
    auto it = token_to_id_.find(w);
    out_tokens->push_back(it == token_to_id_.end() ? 0 : it->second);
  }
  return Error::success();
}

Error BpeTokenizer::decode(const std::vector<std::int64_t>& tokens,
                           std::string* out_text) const {
  if (out_text == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "text out null");
  }
  out_text->clear();
  for (std::int64_t id : tokens) {
    if (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size()) {
      *out_text += id_to_token_[static_cast<std::size_t>(id)];
    }
  }
  return Error::success();
}

}  // namespace tokenizers
}  // namespace uaii
