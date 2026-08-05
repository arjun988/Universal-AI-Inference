#include "uaii/tokenizers/sentencepiece_tokenizer.hpp"

#if defined(UAII_HAVE_SENTENCEPIECE)
#  include <sentencepiece_processor.h>
#endif

namespace uaii {
namespace tokenizers {

struct SentencePieceTokenizer::Impl {
#if defined(UAII_HAVE_SENTENCEPIECE)
  sentencepiece::SentencePieceProcessor sp;
  bool loaded = false;
#endif
};

SentencePieceTokenizer::SentencePieceTokenizer() : impl_(std::make_unique<Impl>()) {}
SentencePieceTokenizer::~SentencePieceTokenizer() = default;

Error SentencePieceTokenizer::load(const std::string& model_path) {
#if defined(UAII_HAVE_SENTENCEPIECE)
  const auto st = impl_->sp.Load(model_path);
  if (!st.ok()) {
    return Error::make(ErrorCode::IoError, "sentencepiece load failed: " + model_path);
  }
  impl_->loaded = true;
  return Error::success();
#else
  (void)model_path;
  return Error::make(ErrorCode::NotImplemented,
                     "SentencePiece requires -DUAII_WITH_SENTENCEPIECE=ON");
#endif
}

Error SentencePieceTokenizer::encode(const std::string& text,
                                     std::vector<std::int64_t>* out_tokens) const {
  if (out_tokens == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "tokens out null");
  }
#if defined(UAII_HAVE_SENTENCEPIECE)
  if (!impl_ || !impl_->loaded) {
    return Error::make(ErrorCode::InvalidArgument, "sentencepiece not loaded");
  }
  std::vector<int> ids;
  const auto st = impl_->sp.Encode(text, &ids);
  if (!st.ok()) return Error::make(ErrorCode::Internal, "sentencepiece encode failed");
  out_tokens->assign(ids.begin(), ids.end());
  return Error::success();
#else
  (void)text;
  (void)out_tokens;
  return Error::make(ErrorCode::NotImplemented, "SentencePiece not built");
#endif
}

Error SentencePieceTokenizer::decode(const std::vector<std::int64_t>& tokens,
                                     std::string* out_text) const {
  if (out_text == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "text out null");
  }
#if defined(UAII_HAVE_SENTENCEPIECE)
  if (!impl_ || !impl_->loaded) {
    return Error::make(ErrorCode::InvalidArgument, "sentencepiece not loaded");
  }
  std::vector<int> ids(tokens.begin(), tokens.end());
  const auto st = impl_->sp.Decode(ids, out_text);
  if (!st.ok()) return Error::make(ErrorCode::Internal, "sentencepiece decode failed");
  return Error::success();
#else
  (void)tokens;
  return Error::make(ErrorCode::NotImplemented, "SentencePiece not built");
#endif
}

}  // namespace tokenizers
}  // namespace uaii
