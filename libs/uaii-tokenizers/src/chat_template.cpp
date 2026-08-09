#include "uaii/tokenizers/chat_template.hpp"

#include <algorithm>
#include <cctype>

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

std::string format_chat_prompt(const std::string& chat_template,
                               const std::string& system,
                               const std::string& user) {
  if (user.find("<|im_start|>") != std::string::npos ||
      user.find("<|start_header_id|>") != std::string::npos ||
      user.find("<start_of_turn>") != std::string::npos ||
      user.find("[INST]") != std::string::npos) {
    return user;
  }

  const std::string t = lower_ascii(chat_template);
  if (t == "llama3") {
    std::string out = "<|begin_of_text|>";
    if (!system.empty()) {
      out += "<|start_header_id|>system<|end_header_id|>\n\n";
      out += system;
      out += "<|eot_id|>";
    }
    out += "<|start_header_id|>user<|end_header_id|>\n\n";
    out += user;
    out += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return out;
  }
  if (t == "gemma") {
    std::string out;
    if (!system.empty()) {
      out += "<start_of_turn>user\n";
      out += system;
      out += "\n\n";
      out += user;
      out += "<end_of_turn>\n";
    } else {
      out += "<start_of_turn>user\n";
      out += user;
      out += "<end_of_turn>\n";
    }
    out += "<start_of_turn>model\n";
    return out;
  }
  if (t == "mistral") {
    std::string body = user;
    if (!system.empty()) body = system + "\n\n" + user;
    return std::string("<s>[INST] ") + body + " [/INST]";
  }

  // ChatML default (Qwen, SmolLM, Phi-3, …)
  std::string out;
  if (!system.empty()) {
    out += "<|im_start|>system\n";
    out += system;
    out += "<|im_end|>\n";
  }
  out += "<|im_start|>user\n";
  out += user;
  out += "<|im_end|>\n";
  out += "<|im_start|>assistant\n";
  return out;
}

std::vector<std::string> default_chat_stops(const std::string& chat_template) {
  const std::string t = lower_ascii(chat_template);
  if (t == "llama3") return {"<|eot_id|>", "<|end_of_text|>"};
  if (t == "gemma") return {"<end_of_turn>"};
  if (t == "mistral") return {"</s>"};
  return {"<|im_end|>", "<|endoftext|>", "</s>"};
}

}  // namespace tokenizers
}  // namespace uaii
