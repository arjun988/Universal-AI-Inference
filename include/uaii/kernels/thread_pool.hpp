#pragma once

#include "uaii/export.hpp"

#include <cstddef>
#include <functional>

namespace uaii {
namespace kernels {

/// Parallel-for over [0, n) using a small portable thread pool (std::thread).
UAII_API void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn);

[[nodiscard]] UAII_API unsigned hardware_concurrency() noexcept;

}  // namespace kernels
}  // namespace uaii
