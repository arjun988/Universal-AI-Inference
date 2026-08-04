#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/tokenizer.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace tokenizers {

/// Whitespace + character fallback tokenizer with optional vocab map.
/// Suitable for demos and plugin-style substitution later.
class UAII_API SimpleTokenizer : public ITokenizer {
 public:
  SimpleTokenizer() = default;

  /// Load vocab from a simple text file: one token per line (id = line index).
  [[nodiscard]] Error load_vocab_file(const std::string& path);

  void set_vocab(std::unordered_map<std::string, std::int64_t> token_to_id);

  [[nodiscard]] std::string name() const override { return "simple"; }

  [[nodiscard]] Error encode(const std::string& text,
                             std::vector<std::int64_t>* out_tokens) override;

  [[nodiscard]] Error decode(const std::vector<std::int64_t>& tokens,
                             std::string* out_text) override;

  [[nodiscard]] std::size_t vocab_size() const noexcept { return id_to_token_.size(); }

  /// Built-in tiny vocab for demos (hello/world/… + specials).
  static SimpleTokenizer demo_vocab();

 private:
  std::unordered_map<std::string, std::int64_t> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::int64_t unk_id_ = 0;
  std::int64_t bos_id_ = 1;
  std::int64_t eos_id_ = 2;
};

}  // namespace tokenizers
}  // namespace uaii
