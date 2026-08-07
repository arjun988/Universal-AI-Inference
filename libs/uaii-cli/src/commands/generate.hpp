#pragma once

#include <string>
#include <vector>

namespace uaii {
namespace cli {

/// One-shot text generation from a GGUF (or tiny --demo) model.
int cmd_generate(const std::vector<std::string>& args);

/// Persistent JSONL chat worker (stdin/stdout) for dashboard warm sessions.
int cmd_chat(const std::vector<std::string>& args);

}  // namespace cli
}  // namespace uaii
