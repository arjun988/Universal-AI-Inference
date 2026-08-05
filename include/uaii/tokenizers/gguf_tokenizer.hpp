#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/tokenizer.hpp"

#include <memory>
#include <string>

namespace uaii {
namespace tokenizers {

/// Best-effort tokenizer from GGUF KV metadata (tokenizer.ggml.tokens / merges).
/// Returns BpeTokenizer when merges are present, otherwise SimpleTokenizer.
[[nodiscard]] UAII_API std::unique_ptr<ITokenizer> load_gguf_tokenizer(const std::string& path,
                                                                         Error* err_out = nullptr);

}  // namespace tokenizers
}  // namespace uaii
