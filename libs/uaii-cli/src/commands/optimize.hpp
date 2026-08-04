#pragma once

#include <string>
#include <vector>

namespace uaii {
namespace cli {

int cmd_profile(const std::vector<std::string>& args);
int cmd_benchmark(const std::vector<std::string>& args);
int cmd_cache(const std::vector<std::string>& args);

}  // namespace cli
}  // namespace uaii
