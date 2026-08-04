#include "uaii/ir/version.hpp"

namespace uaii {
namespace ir {

bool is_compatible(const IrVersion& file_version) noexcept {
  if (file_version.major != kCurrentIrVersion.major) {
    return false;
  }
  return file_version.minor <= kCurrentIrVersion.minor;
}

Error check_compatible(const IrVersion& file_version) {
  if (is_compatible(file_version)) {
    return Error::ok();
  }
  return Error::make(ErrorCode::InvalidArgument,
                     "unsupported UAII IR version " + to_string(file_version) +
                         " (supported major=" + std::to_string(kCurrentIrVersion.major) +
                         ", minor<= " + std::to_string(kCurrentIrVersion.minor) + ")");
}

}  // namespace ir
}  // namespace uaii
