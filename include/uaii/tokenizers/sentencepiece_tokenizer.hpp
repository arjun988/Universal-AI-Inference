#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/tokenizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace uaii {
namespace tokenizers {

/// SentencePiece wrapper (requires UAII_WITH_SENTENCEPIECE + linked lib).
class UAII_API SentencePieceTokenizer : public ITokenizer {
 public:
  SentencePieceTokenizer();
  ~SentencePieceTokenizer() override;

  [[nodiscard]] Error load(const std::string& model_path);

  [[nodiscard]] std::string name() const override { return "sentencepiece"; }

  [[nodiscard]] Error encode(const std::string& text,
                             std::vector<std::int64_t>* out_tokens) const override;
  [[nodiscard]] Error decode(const std::vector<std::int64_t>& tokens,
                             std::string* out_text) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tokenizers
}  // namespace uaii
