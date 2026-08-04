#pragma once

#include "uaii/core/error.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace uaii {

class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  [[nodiscard]] virtual std::string name() const = 0;

  [[nodiscard]] virtual Error encode(const std::string& text,
                                     std::vector<std::int64_t>* out_tokens) = 0;

  [[nodiscard]] virtual Error decode(const std::vector<std::int64_t>& tokens,
                                     std::string* out_text) = 0;
};

}  // namespace uaii
