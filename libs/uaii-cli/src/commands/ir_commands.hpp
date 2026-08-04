#pragma once

#include <string>
#include <vector>

namespace uaii {
namespace cli {

int cmd_validate(const std::vector<std::string>& args);
int cmd_inspect(const std::vector<std::string>& args);
int cmd_graph(const std::vector<std::string>& args);

}  // namespace cli
}  // namespace uaii
