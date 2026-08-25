#include "uaii/tokenizers/chat_template.hpp"

#include <algorithm>
#include <cctype>
#include <variant>

namespace uaii {
namespace tokenizers {
namespace {

std::string lower_ascii(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string kv_string(const loaders::GgufFile& file, const char* key) {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return {};
  if (std::holds_alternative<std::string>(it->second)) return std::get<std::string>(it->second);
  return {};
}

bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

std::string family_of(const std::string& chat_template) {
  const std::string t = lower_ascii(chat_template);
  if (t == "llama3" || t == "llama-3" || t == "llama4") return "llama3";
  if (t == "gemma" || t == "gemma2" || t == "gemma3") return "gemma";
  if (t == "mistral" || t == "mixtral" || t == "llama2") return "mistral";
  if (t == "chatml" || t == "qwen" || t == "qwen2" || t == "qwen3") return "chatml";
  if (contains(chat_template, "<|start_header_id|>")) return "llama3";
  if (contains(chat_template, "<start_of_turn>")) return "gemma";
  if (contains(chat_template, "[INST]")) return "mistral";
  if (contains(chat_template, "<|im_start|>")) return "chatml";
  return t.empty() ? "chatml" : t;
}

bool looks_preformatted(const std::string& text) {
  return contains(text, "<|im_start|>") || contains(text, "<|start_header_id|>") ||
         contains(text, "<start_of_turn>") || contains(text, "[INST]");
}

}  // namespace

std::string detect_chat_template(const loaders::GgufFile& file, const std::string& arch) {
  const std::string jinja = kv_string(file, "tokenizer.chat_template");
  if (contains(jinja, "<|start_header_id|>") || contains(jinja, "llama3")) {
    return "llama3";
  }
  if (contains(jinja, "<start_of_turn>") || contains(jinja, "gemma")) {
    return "gemma";
  }
  if (contains(jinja, "[INST]") || contains(jinja, "mistral")) {
    return "mistral";
  }
  if (contains(jinja, "<|im_start|>")) {
    return "chatml";
  }

  const std::string a = lower_ascii(arch);
  if (a == "llama" || a == "llama3" || a == "llama4") return "llama3";
  if (a == "gemma" || a == "gemma2" || a == "gemma3") return "gemma";
  if (a == "mistral" || a == "mixtral" || a == "llama2") return "mistral";
  if (a == "qwen" || a == "qwen2" || a == "qwen2moe" || a == "qwen3" || a == "chatglm" ||
      a == "phi3" || a == "phi" || a == "smollm") {
    return "chatml";
  }
  return "chatml";
}

std::string format_chat_messages(const std::string& chat_template,
                                 const std::vector<ChatMessage>& messages) {
  if (messages.size() == 1 && looks_preformatted(messages[0].content)) {
    return messages[0].content;
  }

  const std::string fam = family_of(chat_template);
  std::string system;
  std::vector<ChatMessage> turns;
  turns.reserve(messages.size());
  for (const auto& m : messages) {
    const std::string role = lower_ascii(m.role);
    if (role == "system") {
      if (system.empty()) system = m.content;
      else system += "\n" + m.content;
      continue;
    }
    ChatMessage t;
    t.role = (role == "assistant" || role == "model") ? "assistant" : "user";
    t.content = m.content;
    turns.push_back(std::move(t));
  }

  if (fam == "llama3") {
    std::string out = "<|begin_of_text|>";
    if (!system.empty()) {
      out += "<|start_header_id|>system<|end_header_id|>\n\n";
      out += system;
      out += "<|eot_id|>";
    }
    for (const auto& m : turns) {
      const char* tag = m.role == "assistant" ? "assistant" : "user";
      out += "<|start_header_id|>";
      out += tag;
      out += "<|end_header_id|>\n\n";
      out += m.content;
      out += "<|eot_id|>";
    }
    out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return out;
  }

  if (fam == "gemma") {
    // Gemma: no dedicated system turn; fold system into the first user message.
    // Keep assistant history (continue-on-assistant is a dashboard bug).
    std::string out;
    bool first_user = true;
    for (const auto& m : turns) {
      if (m.role == "assistant") {
        out += "<start_of_turn>model\n";
        out += m.content;
        out += "<end_of_turn>\n";
        continue;
      }
      out += "<start_of_turn>user\n";
      if (first_user && !system.empty()) {
        out += system;
        out += "\n\n";
      }
      out += m.content;
      out += "<end_of_turn>\n";
      first_user = false;
    }
    if (turns.empty() && !system.empty()) {
      out += "<start_of_turn>user\n";
      out += system;
      out += "<end_of_turn>\n";
    }
    out += "<start_of_turn>model\n";
    return out;
  }

  if (fam == "mistral") {
    std::string out = "<s>";
    std::string pending = system.empty() ? std::string() : system + "\n\n";
    bool opened = false;
    for (const auto& m : turns) {
      if (m.role == "user") {
        pending += m.content;
        out += "[INST] ";
        out += pending;
        out += " [/INST]";
        pending.clear();
        opened = true;
      } else {
        if (!opened) {
          out += "[INST] ";
          if (!pending.empty()) out += pending;
          out += " [/INST]";
          pending.clear();
          opened = true;
        }
        out += " ";
        out += m.content;
        out += "</s>";
        opened = false;
      }
    }
    return out;
  }

  std::string out;
  if (!system.empty()) {
    out += "<|im_start|>system\n";
    out += system;
    out += "<|im_end|>\n";
  }
  for (const auto& m : turns) {
    const char* tag = m.role == "assistant" ? "assistant" : "user";
    out += "<|im_start|>";
    out += tag;
    out += "\n";
    out += m.content;
    out += "<|im_end|>\n";
  }
  out += "<|im_start|>assistant\n";
  return out;
}

std::string format_chat_prompt(const std::string& chat_template,
                               const std::string& system,
                               const std::string& user) {
  if (looks_preformatted(user)) {
    return user;
  }
  std::vector<ChatMessage> msgs;
  if (!system.empty()) msgs.push_back({"system", system});
  msgs.push_back({"user", user});
  return format_chat_messages(chat_template, msgs);
}

std::vector<std::string> default_chat_stops(const std::string& chat_template) {
  const std::string t = family_of(chat_template);
  if (t == "llama3") return {"<|eot_id|>", "<|end_of_text|>"};
  if (t == "gemma") return {"<end_of_turn>"};
  if (t == "mistral") return {"</s>"};
  return {"<|im_end|>", "<|endoftext|>", "</s>"};
}

}  // namespace tokenizers
}  // namespace uaii
