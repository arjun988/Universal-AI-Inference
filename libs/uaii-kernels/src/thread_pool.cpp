#include "uaii/kernels/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

namespace uaii {
namespace kernels {
namespace {

std::atomic<unsigned> g_thread_limit{0};

}  // namespace

void set_thread_limit(unsigned n) noexcept {
  g_thread_limit.store(n, std::memory_order_relaxed);
}

unsigned hardware_concurrency() noexcept {
  unsigned n = 0;
  const char* env = std::getenv("UAII_NUM_THREADS");
  if (env && env[0] != '\0') {
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    if (end != env && v > 0 && v < 4096) {
      n = static_cast<unsigned>(v);
    }
  }
  if (n == 0) {
    n = std::thread::hardware_concurrency();
    if (n == 0) n = 1u;
  }
  const unsigned cap = g_thread_limit.load(std::memory_order_relaxed);
  if (cap > 0 && n > cap) n = cap;
  return n == 0 ? 1u : n;
}

void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
  if (n == 0 || !fn) return;
  if (n == 1) {
    fn(0);
    return;
  }
  const unsigned workers =
      static_cast<unsigned>(std::min<std::size_t>(n, hardware_concurrency()));
  if (workers <= 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }

  std::atomic<std::size_t> next{0};
  auto body = [&]() {
    for (;;) {
      const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) break;
      fn(i);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (unsigned t = 1; t < workers; ++t) {
    threads.emplace_back(body);
  }
  body();
  for (auto& th : threads) th.join();
}

}  // namespace kernels
}  // namespace uaii
