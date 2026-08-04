#include "uaii/core/error.hpp"

namespace uaii {

const char* to_string(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Ok: return "ok";
    case ErrorCode::InvalidArgument: return "invalid_argument";
    case ErrorCode::NotFound: return "not_found";
    case ErrorCode::IoError: return "io_error";
    case ErrorCode::ConfigError: return "config_error";
    case ErrorCode::PluginError: return "plugin_error";
    case ErrorCode::AbiMismatch: return "abi_mismatch";
    case ErrorCode::NotImplemented: return "not_implemented";
    case ErrorCode::Internal: return "internal";
  }
  return "unknown";
}

std::string Error::to_string() const {
  if (ok()) {
    return "ok";
  }
  std::string out = uaii::to_string(code_);
  if (!message_.empty()) {
    out += ": ";
    out += message_;
  }
  return out;
}

}  // namespace uaii
