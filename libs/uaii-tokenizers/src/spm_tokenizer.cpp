#include "uaii/tokenizers/spm_tokenizer.hpp"

#include <algorithm>
#include <sstream>

namespace uaii {
namespace tokenizers {
namespace {

constexpr char32_t kSpmSpace = 0x2581;  // ▁

std::string utf8_encode(char32_t cp) {
  std::string out;
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

std::string spm_preprocess(const std::string& text) {
  std::string out = utf8_encode(kSpmSpace);
  for (unsigned char c : text) {
    if (c == ' ') {
      out += utf8_encode(kSpmSpace);
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

bool is_special_piece(const std::string& piece) {
  return !piece.empty() && piece.front() == '<' && piece.back() == '>';
}

}  // namespace

Error SpmTokenizer::load_from_gguf_tokens(const std::vector<std::string>& tokens,
                                          std::int64_t bos_id,
                                          std::int64_t eos_id,
                                          std::int64_t unk_id,
                                          bool add_bos,
                                          bool add_eos) {
  if (tokens.empty()) {
    return Error::make(ErrorCode::InvalidArgument, "empty SPM token list");
  }
  token_to_id_.clear();
  id_to_token_ = tokens;
  max_piece_len_ = 0;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    token_to_id_[tokens[i]] = static_cast<std::int64_t>(i);
    max_piece_len_ = std::max(max_piece_len_, tokens[i].size());
  }
  bos_id_ = bos_id;
  eos_id_ = eos_id;
  unk_id_ = unk_id >= 0 ? unk_id : 0;
  add_bos_ = add_bos;
  add_eos_ = add_eos;
  return Error::success();
}

Error SpmTokenizer::encode(const std::string& text,
                           std::vector<std::int64_t>* out_tokens) const {
  if (out_tokens == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tokens out null");
  }
  out_tokens->clear();
  if (add_bos_ && bos_id_ >= 0) out_tokens->push_back(bos_id_);

  const std::string normalized = spm_preprocess(text);
  std::size_t i = 0;
  while (i < normalized.size()) {
    std::size_t best_len = 0;
    std::int64_t best_id = unk_id_;
    const std::size_t max_len =
        std::min(max_piece_len_, normalized.size() - i);
    for (std::size_t len = max_len; len > 0; --len) {
      const std::string piece = normalized.substr(i, len);
      const auto it = token_to_id_.find(piece);
      if (it != token_to_id_.end()) {
        best_len = len;
        best_id = it->second;
        break;
      }
    }
    if (best_len == 0) {
      out_tokens->push_back(unk_id_);
      ++i;
      continue;
    }
    out_tokens->push_back(best_id);
    i += best_len;
  }

  if (add_eos_ && eos_id_ >= 0) out_tokens->push_back(eos_id_);
  return Error::success();
}

Error SpmTokenizer::decode(const std::vector<std::int64_t>& tokens,
                           std::string* out_text) const {
  if (out_text == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "text out null");
  }
  std::string raw;
  for (std::int64_t id : tokens) {
    if (id == bos_id_ || id == eos_id_) continue;
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) continue;
    const std::string& piece = id_to_token_[static_cast<std::size_t>(id)];
    if (is_special_piece(piece)) continue;
    raw += piece;
  }

  const std::string spm_space = utf8_encode(kSpmSpace);
  std::string out;
  std::size_t pos = 0;
  while (pos < raw.size()) {
    if (raw.compare(pos, spm_space.size(), spm_space) == 0) {
      out.push_back(' ');
      pos += spm_space.size();
    } else {
      out.push_back(raw[pos++]);
    }
  }
  if (!out.empty() && out.front() == ' ') out.erase(0, 1);
  *out_text = std::move(out);
  return Error::success();
}

}  // namespace tokenizers
}  // namespace uaii
