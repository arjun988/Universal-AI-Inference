#include "uaii/core/config.hpp"
#include "uaii/core/error.hpp"
#include "uaii/core/log.hpp"
#include "uaii/version.hpp"

#include <cstdlib>
#include <iostream>

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  expect(uaii::version().major == 0, "version major");
  expect(uaii::Error::success().ok(), "Error::ok");
  expect(!uaii::Error::make(uaii::ErrorCode::Internal, "x").ok(), "Error failure");

  uaii::Config cfg;
  auto err = cfg.parse(R"(
[log]
level = "debug"
color = "true"

[plugin]
dirs = "a,b;c"
)",
                       "smoke");
  expect(err.ok(), "config parse");
  expect(cfg.get_string("log.level") == "debug", "log.level");
  expect(cfg.get_bool("log.color", false), "log.color");
  auto dirs = uaii::Config::split_list(cfg.get_string("plugin.dirs"));
  expect(dirs.size() == 3, "plugin.dirs split");

  uaii::log::Level level = uaii::log::Level::Off;
  expect(uaii::log::parse_level("INFO", &level), "parse_level");
  expect(level == uaii::log::Level::Info, "parse_level value");

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_smoke_test: OK\n";
  return EXIT_SUCCESS;
}
