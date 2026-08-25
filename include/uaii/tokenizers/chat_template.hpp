#pragma once

#include "uaii/export.hpp"
#include "uaii/loaders/gguf.hpp"

#include <string>
#include <vector>

namespace uaii {
namespace tokenizers {

struct ChatMessage {
  std::string role;
  std::string content;
};

/// Detect chat formatting from GGUF metadata / architecture.
[[nodiscard]] UAII_API std::string detect_chat_template(const loaders::GgufFile& file,
                                                        const std::string& arch);

/// Multi-turn family formatter (Llama-3 / Gemma / Mistral / ChatML).
/// This is the documented equivalent of the shipped Jinja `chat_template`, not a
/// Jinja interpreter. Tool-calling and Llama-4 extras are not executed.
[[nodiscard]] UAII_API std::string format_chat_messages(const std::string& chat_template,
                                                        const std::vector<ChatMessage>& messages);

/// Format a single user turn (optional system) for inference.
[[nodiscard]] UAII_API std::string format_chat_prompt(const std::string& chat_template,
                                                      const std::string& system,
                                                      const std::string& user);

/// Default stop strings for a template (may be empty).
[[nodiscard]] UAII_API std::vector<std::string> default_chat_stops(
    const std::string& chat_template);

}  // namespace tokenizers
}  // namespace uaii
