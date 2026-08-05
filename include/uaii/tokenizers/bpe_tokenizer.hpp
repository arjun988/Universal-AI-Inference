#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/tokenizer.hpp"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uaii {
namespace tokenizers {

/// GPT-2 style BPE (vocab.json + merges.txt).
class UAII_API BpeTokenizer : public ITokenizer {
 public:
  [[nodiscard]] Error load(const std::string& vocab_json_path,
                           const std::string& merges_path);

  /// In-memory vocab (id = index) + merge lines ("a b").
  [[nodiscard]] Error load_from_vocab_merges(const std::vector<std::string>& tokens,
                                             const std::vector<std::string>& merge_lines);

  [[nodiscard]] std::string name() const override { return "bpe"; }

  [[nodiscard]] Error encode(const std::string& text,
                             std::vector<std::int64_t>* out_tokens) const override;
  [[nodiscard]] Error decode(const std::vector<std::int64_t>& tokens,
                             std::string* out_text) const override;

 private:
  std::unordered_map<std::string, std::int64_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::vector<std::pair<std::string, std::string>> merges_;
};

}  // namespace tokenizers
}  // namespace uaii
