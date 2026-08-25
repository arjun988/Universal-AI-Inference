#include "uaii/tokenizers/bpe_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace uaii {
namespace tokenizers {
namespace {

// GPT-2 / Qwen printable byte ↔ unicode mapping (huggingface tokenizers).
void build_byte_maps(std::array<char32_t, 256>* byte_encoder,
                     std::unordered_map<char32_t, unsigned char>* byte_decoder) {
  std::vector<int> bs;
  for (int i = 33; i <= 126; ++i) bs.push_back(i);    // '!' .. '~'
  for (int i = 161; i <= 172; ++i) bs.push_back(i);   // ¡ .. ¬
  for (int i = 174; i <= 255; ++i) bs.push_back(i);   // ® .. ÿ
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (std::size_t i = 0; i < bs.size(); ++i) {
    (*byte_encoder)[static_cast<std::size_t>(bs[i])] = static_cast<char32_t>(cs[i]);
    (*byte_decoder)[static_cast<char32_t>(cs[i])] = static_cast<unsigned char>(bs[i]);
  }
}

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

bool utf8_next(const std::string& s, std::size_t* i, char32_t* cp) {
  if (*i >= s.size()) return false;
  const auto c0 = static_cast<unsigned char>(s[*i]);
  if (c0 < 0x80) {
    *cp = c0;
    ++(*i);
    return true;
  }
  if ((c0 & 0xE0) == 0xC0 && *i + 1 < s.size()) {
    *cp = (static_cast<char32_t>(c0 & 0x1F) << 6) |
          (static_cast<unsigned char>(s[*i + 1]) & 0x3F);
    *i += 2;
    return true;
  }
  if ((c0 & 0xF0) == 0xE0 && *i + 2 < s.size()) {
    *cp = (static_cast<char32_t>(c0 & 0x0F) << 12) |
          ((static_cast<unsigned char>(s[*i + 1]) & 0x3F) << 6) |
          (static_cast<unsigned char>(s[*i + 2]) & 0x3F);
    *i += 3;
    return true;
  }
  if ((c0 & 0xF8) == 0xF0 && *i + 3 < s.size()) {
    *cp = (static_cast<char32_t>(c0 & 0x07) << 18) |
          ((static_cast<unsigned char>(s[*i + 1]) & 0x3F) << 12) |
          ((static_cast<unsigned char>(s[*i + 2]) & 0x3F) << 6) |
          (static_cast<unsigned char>(s[*i + 3]) & 0x3F);
    *i += 4;
    return true;
  }
  *cp = c0;
  ++(*i);
  return true;
}

bool is_unicode_letter(char32_t c) {
  if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
  if (c >= 0x00C0 && c <= 0x02AF) return true;
  if (c >= 0x0370 && c <= 0x03FF) return true;
  if (c >= 0x0400 && c <= 0x052F) return true;
  if (c >= 0x0590 && c <= 0x05FF) return true;
  if (c >= 0x0600 && c <= 0x06FF) return true;
  if (c >= 0x0900 && c <= 0x097F) return true;
  if (c >= 0x0E00 && c <= 0x0E7F) return true;
  if (c >= 0x1E00 && c <= 0x1EFF) return true;
  if (c >= 0x2C60 && c <= 0x2C7F) return true;
  if (c >= 0x3040 && c <= 0x30FF) return true;
  if (c >= 0x3400 && c <= 0x9FFF) return true;
  if (c >= 0xAC00 && c <= 0xD7AF) return true;
  if (c >= 0xF900 && c <= 0xFAFF) return true;
  if (c >= 0xFF21 && c <= 0xFF3A) return true;
  if (c >= 0xFF41 && c <= 0xFF5A) return true;
  return false;
}

bool is_unicode_digit(char32_t c) {
  return (c >= '0' && c <= '9') || (c >= 0xFF10 && c <= 0xFF19);
}

bool is_unicode_space(char32_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == 0x00A0 ||
         c == 0x3000;
}

std::size_t contraction_len(const std::vector<char32_t>& cps, std::size_t i) {
  if (i >= cps.size() || cps[i] != '\'') return 0;
  auto eq = [](char32_t a, char b) {
    if (a >= 'A' && a <= 'Z') a = static_cast<char32_t>(a - 'A' + 'a');
    return a == static_cast<char32_t>(b);
  };
  if (i + 1 < cps.size() &&
      (eq(cps[i + 1], 's') || eq(cps[i + 1], 't') || eq(cps[i + 1], 'm') || eq(cps[i + 1], 'd'))) {
    return 2;
  }
  if (i + 2 < cps.size() &&
      ((eq(cps[i + 1], 'r') && eq(cps[i + 2], 'e')) || (eq(cps[i + 1], 'v') && eq(cps[i + 2], 'e')) ||
       (eq(cps[i + 1], 'l') && eq(cps[i + 2], 'l')))) {
    return 3;
  }
  return 0;
}

std::string utf8_from_cps(const std::vector<char32_t>& cps, std::size_t begin, std::size_t end) {
  std::string out;
  for (std::size_t i = begin; i < end && i < cps.size(); ++i) out += utf8_encode(cps[i]);
  return out;
}

std::vector<std::string> gpt2_regex_split(const std::string& text) {
  std::vector<char32_t> cps;
  std::size_t i = 0;
  char32_t cp = 0;
  while (utf8_next(text, &i, &cp)) cps.push_back(cp);
  std::vector<std::string> pieces;
  std::size_t p = 0;
  while (p < cps.size()) {
    if (const std::size_t n = contraction_len(cps, p)) {
      pieces.push_back(utf8_from_cps(cps, p, p + n));
      p += n;
      continue;
    }
    const bool lead_space = is_unicode_space(cps[p]);
    std::size_t q = p;
    if (lead_space && q + 1 < cps.size() && !is_unicode_space(cps[q + 1])) {
      ++q;  // optional single leading space (the ` ?` in the GPT-2 regex)
    }
    if (q < cps.size() && is_unicode_letter(cps[q])) {
      while (q < cps.size() && is_unicode_letter(cps[q])) ++q;
      pieces.push_back(utf8_from_cps(cps, p, q));
      p = q;
      continue;
    }
    if (q < cps.size() && is_unicode_digit(cps[q])) {
      while (q < cps.size() && is_unicode_digit(cps[q])) ++q;
      pieces.push_back(utf8_from_cps(cps, p, q));
      p = q;
      continue;
    }
    if (q < cps.size() && !is_unicode_space(cps[q]) && !is_unicode_letter(cps[q]) &&
        !is_unicode_digit(cps[q])) {
      while (q < cps.size() && !is_unicode_space(cps[q]) && !is_unicode_letter(cps[q]) &&
             !is_unicode_digit(cps[q])) {
        ++q;
      }
      pieces.push_back(utf8_from_cps(cps, p, q));
      p = q;
      continue;
    }
    // whitespace run
    q = p;
    while (q < cps.size() && is_unicode_space(cps[q])) ++q;
    pieces.push_back(utf8_from_cps(cps, p, q));
    p = q;
  }
  return pieces;
}

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

BpeTokenizer::BpeTokenizer() {
  build_byte_maps(&byte_encoder_, &byte_decoder_);
}

void BpeTokenizer::ensure_specials() {
  // Keep explicitly registered specials; re-scan <|...|> plus common control tokens.
  std::vector<std::string> extra;
  for (const auto& kv : token_to_id_) {
    const auto& t = kv.first;
    if (t.size() >= 3 && t.rfind("<|", 0) == 0 && t.size() >= 5 &&
        t.compare(t.size() - 2, 2, "|>") == 0) {
      extra.push_back(t);
    }
  }
  for (const auto& t : extra) {
    if (std::find(special_tokens_.begin(), special_tokens_.end(), t) == special_tokens_.end()) {
      special_tokens_.push_back(t);
    }
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
}

void BpeTokenizer::register_special(const std::string& token) {
  if (token.empty()) return;
  if (std::find(special_tokens_.begin(), special_tokens_.end(), token) == special_tokens_.end()) {
    special_tokens_.push_back(token);
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
}

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
  ensure_specials();
  pretokenize_ = true;
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
  ensure_specials();
  return Error::success();
}

Error BpeTokenizer::bpe_encode_piece(const std::string& piece,
                                     std::vector<std::int64_t>* out_tokens) const {
  // Exact vocab hit (special tokens, whole words already in vocab as-is).
  auto exact = token_to_id_.find(piece);
  if (exact != token_to_id_.end()) {
    out_tokens->push_back(exact->second);
    return Error::success();
  }

  // GPT-2: map UTF-8 bytes → unicode chars, then apply merges.
  std::vector<std::string> word;
  word.reserve(piece.size());
  for (unsigned char c : piece) {
    word.push_back(utf8_encode(byte_encoder_[c]));
  }
  if (word.empty()) return Error::success();

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

Error BpeTokenizer::encode(const std::string& text,
                           std::vector<std::int64_t>* out_tokens) const {
  if (out_tokens == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tokens out null");
  }
  out_tokens->clear();

  if (add_bos_ && bos_id_ >= 0) {
    out_tokens->push_back(bos_id_);
  }

  std::string work = text;
  if (!metaspace_.empty()) {
    std::string mapped;
    mapped.reserve(work.size() + metaspace_.size());
    if (!work.empty()) mapped += metaspace_;
    for (char c : work) {
      if (c == ' ') mapped += metaspace_;
      else mapped.push_back(c);
    }
    work = std::move(mapped);
  }
  if (pretokenize_ && add_prefix_space_ && !work.empty() && work[0] != ' ') {
    bool starts_special = false;
    for (const auto& sp : special_tokens_) {
      if (work.compare(0, sp.size(), sp) == 0) {
        starts_special = true;
        break;
      }
    }
    if (!starts_special) work.insert(work.begin(), ' ');
  }

  // Split out special tokens so they map to single ids.
  std::size_t i = 0;
  while (i < work.size()) {
    bool hit = false;
    for (const auto& sp : special_tokens_) {
      if (work.compare(i, sp.size(), sp) == 0) {
        auto it = token_to_id_.find(sp);
        if (it != token_to_id_.end()) {
          out_tokens->push_back(it->second);
          i += sp.size();
          hit = true;
          break;
        }
      }
    }
    if (hit) continue;

    std::size_t j = i + 1;
    while (j < work.size()) {
      bool at_special = false;
      for (const auto& sp : special_tokens_) {
        if (work.compare(j, sp.size(), sp) == 0 && token_to_id_.count(sp)) {
          at_special = true;
          break;
        }
      }
      if (at_special) break;
      ++j;
    }
    const std::string chunk = work.substr(i, j - i);
    if (pretokenize_) {
      for (const auto& piece : gpt2_regex_split(chunk)) {
        Error err = bpe_encode_piece(piece, out_tokens);
        if (!err.ok()) return err;
      }
    } else {
      Error err = bpe_encode_piece(chunk, out_tokens);
      if (!err.ok()) return err;
    }
    i = j;
  }
  return Error::success();
}

Error BpeTokenizer::decode(const std::vector<std::int64_t>& tokens,
                           std::string* out_text) const {
  if (out_text == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "text out null");
  }
  out_text->clear();
  std::string merged;
  for (std::int64_t id : tokens) {
    if (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size()) {
      const std::string& tok = id_to_token_[static_cast<std::size_t>(id)];
      bool special = false;
      for (const auto& sp : special_tokens_) {
        if (tok == sp) {
          special = true;
          break;
        }
      }
      if (!special && tok.size() >= 3 && tok.rfind("<|", 0) == 0 &&
          tok.compare(tok.size() - 2, 2, "|>") == 0) {
        special = true;
      }
      if (special) {
        if (!merged.empty()) {
          // flush pending byte-decoded text
          std::string chunk;
          std::size_t p = 0;
          char32_t cp = 0;
          while (utf8_next(merged, &p, &cp)) {
            auto it = byte_decoder_.find(cp);
            if (it != byte_decoder_.end()) chunk.push_back(static_cast<char>(it->second));
          }
          *out_text += chunk;
          merged.clear();
        }
        *out_text += tok;
        continue;
      }
      merged += tok;
    }
  }
  if (!merged.empty()) {
    std::string chunk;
    std::size_t p = 0;
    char32_t cp = 0;
    while (utf8_next(merged, &p, &cp)) {
      auto it = byte_decoder_.find(cp);
      if (it != byte_decoder_.end()) {
        chunk.push_back(static_cast<char>(it->second));
      }
    }
    *out_text += chunk;
  }
  return Error::success();
}

}  // namespace tokenizers
}  // namespace uaii
