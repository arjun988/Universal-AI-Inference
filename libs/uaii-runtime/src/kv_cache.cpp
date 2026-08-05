#include "uaii/runtime/kv_cache.hpp"

// KvCache methods are inline in the header so kernels can use them without a
// link cycle (runtime → kernels). This TU anchors the runtime module.
namespace uaii {
namespace runtime {
namespace {
[[maybe_unused]] const int kKvCacheTuAnchor = 0;
}
}  // namespace runtime
}  // namespace uaii
