#include "uaii/tokenizers/simple_tokenizer.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace uaii {
namespace tokenizers {

Error SimpleTokenizer::load_vocab_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return Error::make(ErrorCode::NotFound, "vocab file not found: " + path);
  }
  token_to_id_.clear();
  id_to_token_.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto id = static_cast<std::int64_t>(id_to_token_.size());
    token_to_id_[line] = id;
    id_to_token_.push_back(line);
  }
  if (id_to_token_.empty()) {
    return Error::make(ErrorCode::InvalidArgument, "empty vocab");
  }
  unk_id_ = token_to_id_.count("<unk>") ? token_to_id_["<unk>"] : 0;
  bos_id_ = token_to_id_.count("<bos>") ? token_to_id_["<bos>"] : unk_id_;
  eos_id_ = token_to_id_.count("<eos>") ? token_to_id_["<eos>"] : unk_id_;
  return Error::success();
}

void SimpleTokenizer::set_vocab(std::unordered_map<std::string, std::int64_t> token_to_id) {
  token_to_id_ = std::move(token_to_id);
  std::int64_t max_id = -1;
  for (const auto& kv : token_to_id_) {
    if (kv.second > max_id) max_id = kv.second;
  }
  id_to_token_.assign(static_cast<std::size_t>(max_id + 1), "<unk>");
  for (const auto& kv : token_to_id_) {
    if (kv.second >= 0) {
      id_to_token_[static_cast<std::size_t>(kv.second)] = kv.first;
    }
  }
  unk_id_ = token_to_id_.count("<unk>") ? token_to_id_["<unk>"] : 0;
  bos_id_ = token_to_id_.count("<bos>") ? token_to_id_["<bos>"] : unk_id_;
  eos_id_ = token_to_id_.count("<eos>") ? token_to_id_["<eos>"] : unk_id_;
}

SimpleTokenizer SimpleTokenizer::demo_vocab() {
  SimpleTokenizer t;
  t.set_vocab({
      {"<unk>", 0},
      {"<bos>", 1},
      {"<eos>", 2},
      {"hello", 3},
      {"world", 4},
      {"uaii", 5},
      {"runtime", 6},
      {"the", 7},
      {"a", 8},
      {"model", 9},
  });
  return t;
}

Error SimpleTokenizer::encode(const std::string& text,
                              std::vector<std::int64_t>* out_tokens) const {
  if (out_tokens == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tokens out null");
  }
  out_tokens->clear();
  out_tokens->push_back(bos_id_);

  std::string token;
  auto flush = [&]() {
    if (token.empty()) return;
    auto it = token_to_id_.find(token);
    out_tokens->push_back(it == token_to_id_.end() ? unk_id_ : it->second);
    token.clear();
  };

  for (char c : text) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      flush();
    } else {
      token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  flush();
  out_tokens->push_back(eos_id_);
  return Error::success();
}

Error SimpleTokenizer::decode(const std::vector<std::int64_t>& tokens,
                              std::string* out_text) const {
  if (out_text == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "text out null");
  }
  std::ostringstream oss;
  bool first = true;
  for (std::int64_t id : tokens) {
    if (id == bos_id_ || id == eos_id_) continue;
    std::string piece = "<unk>";
    if (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size()) {
      piece = id_to_token_[static_cast<std::size_t>(id)];
    }
    if (!first) oss << ' ';
    first = false;
    oss << piece;
  }
  *out_text = oss.str();
  return Error::success();
}

}  // namespace tokenizers
}  // namespace uaii
