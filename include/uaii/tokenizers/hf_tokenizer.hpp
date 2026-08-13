#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/tokenizer.hpp"

#include <memory>
#include <string>

namespace uaii {
namespace tokenizers {

/// Load tokenizer from an HF model directory (tokenizer.json, vocab+merges, or .model).
[[nodiscard]] UAII_API std::unique_ptr<ITokenizer> load_hf_tokenizer(
    const std::string& model_dir, Error* err);

/// Load chat_template string from tokenizer_config.json if present.
[[nodiscard]] UAII_API std::string load_hf_chat_template(const std::string& model_dir);

/// Detect chat template family name (llama3/gemma/mistral/chatml) from Jinja or arch.
[[nodiscard]] UAII_API std::string detect_hf_chat_template_name(
    const std::string& jinja_or_empty, const std::string& architecture);

}  // namespace tokenizers
}  // namespace uaii
