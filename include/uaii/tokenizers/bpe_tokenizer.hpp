#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/tokenizer.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uaii {
namespace tokenizers {

/// GPT-2 / Qwen style BPE (vocab + merges) with byte↔unicode mapping.
class UAII_API BpeTokenizer : public ITokenizer {
 public:
  BpeTokenizer();

  [[nodiscard]] Error load(const std::string& vocab_json_path,
                           const std::string& merges_path);

  /// In-memory vocab (id = index) + merge lines ("a b").
  [[nodiscard]] Error load_from_vocab_merges(const std::vector<std::string>& tokens,
                                             const std::vector<std::string>& merge_lines);

  /// Register a special token that must encode/decode as a single id (added_tokens).
  void register_special(const std::string& token);

  /// Hugging Face ByteLevel `add_prefix_space`.
  void set_add_prefix_space(bool enabled) noexcept { add_prefix_space_ = enabled; }

  /// Prepend BOS id when tokenizer_config.json has add_bos_token.
  void set_add_bos(bool enabled, std::int64_t bos_id = -1) noexcept {
    add_bos_ = enabled;
    bos_id_ = bos_id;
  }

  /// Metaspace replacement (Llama-2 ▁). Empty = off.
  void set_metaspace(std::string replacement) { metaspace_ = std::move(replacement); }

  /// GPT-2 / Llama-3 byte-level regex split before BPE (default off for GGUF).
  void set_byte_level_pretokenize(bool enabled) noexcept { pretokenize_ = enabled; }

  [[nodiscard]] std::string name() const override { return "bpe"; }

  [[nodiscard]] Error encode(const std::string& text,
                             std::vector<std::int64_t>* out_tokens) const override;
  [[nodiscard]] Error decode(const std::vector<std::int64_t>& tokens,
                             std::string* out_text) const override;

 private:
  void ensure_specials();
  [[nodiscard]] Error bpe_encode_piece(const std::string& piece,
                                       std::vector<std::int64_t>* out_tokens) const;

  std::unordered_map<std::string, std::int64_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::vector<std::pair<std::string, std::string>> merges_;
  std::vector<std::string> special_tokens_;
  std::array<char32_t, 256> byte_encoder_{};
  std::unordered_map<char32_t, unsigned char> byte_decoder_;
  bool add_prefix_space_ = false;
  bool pretokenize_ = false;
  bool add_bos_ = false;
  std::int64_t bos_id_ = -1;
  std::string metaspace_;
};

}  // namespace tokenizers
}  // namespace uaii
