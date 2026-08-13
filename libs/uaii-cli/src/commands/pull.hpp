#pragma once

#include <string>
#include <vector>

namespace uaii {
namespace cli {

/// Download a Hugging Face repo snapshot into a local cache directory.
int cmd_pull(const std::vector<std::string>& args);

}  // namespace cli
}  // namespace uaii
