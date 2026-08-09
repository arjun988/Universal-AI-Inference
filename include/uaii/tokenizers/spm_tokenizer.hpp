#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/tokenizer.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace tokenizers {

/// SentencePiece-style greedy tokenizer from GGUF `tokenizer.ggml.tokens` (no merges).
/// Used by Llama-2, Mistral, TinyLlama, and most non-BPE GGUF exports.
class UAII_API SpmTokenizer : public ITokenizer {
 public:
  SpmTokenizer() = default;

  [[nodiscard]] Error load_from_gguf_tokens(const std::vector<std::string>& tokens,
                                            std::int64_t bos_id,
                                            std::int64_t eos_id,
                                            std::int64_t unk_id,
                                            bool add_bos,
                                            bool add_eos);

  [[nodiscard]] std::string name() const override { return "spm"; }

  [[nodiscard]] Error encode(const std::string& text,
                             std::vector<std::int64_t>* out_tokens) const override;

  [[nodiscard]] Error decode(const std::vector<std::int64_t>& tokens,
                             std::string* out_text) const override;

 private:
  std::unordered_map<std::string, std::int64_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::size_t max_piece_len_ = 0;
  std::int64_t bos_id_ = -1;
  std::int64_t eos_id_ = -1;
  std::int64_t unk_id_ = 0;
  bool add_bos_ = false;
  bool add_eos_ = false;
};

}  // namespace tokenizers
}  // namespace uaii
