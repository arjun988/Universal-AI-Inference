#include "uaii/runtime/kv_cache.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

uaii::kernels::TensorView view_f32(std::vector<float>& buf, std::int64_t d) {
  static thread_local std::int64_t shape[2];
  shape[0] = 1;
  shape[1] = d;
  uaii::kernels::TensorView v;
  v.dtype = uaii::DType::F32;
  v.shape = shape;
  v.rank = 2;
  v.data = buf.data();
  v.nbytes = buf.size() * sizeof(float);
  return v;
}

}  // namespace

int main() {
  using namespace uaii::runtime;
  KvCache cache;
  const uaii::Error cerr =
      cache.configure(1, 1, /*max_seq=*/8, /*k_dim=*/4, /*v_dim=*/4, /*n_heads=*/2, /*sink=*/2);
  expect(cerr.ok(), "configure");
  expect(cache.rope_pos() == 0, "rope starts at 0");

  std::vector<float> k(4, 1.f);
  std::vector<float> v(4, 2.f);
  for (int t = 0; t < 12; ++t) {
    k[0] = static_cast<float>(t);
    const auto kv = view_f32(k, 4);
    const auto vv = view_f32(v, 4);
    expect(cache.append_layer(0, kv, vv).ok(), "append");
    cache.commit_step();
  }
  expect(cache.past_len() == 8, "window caps occupancy");
  expect(cache.rope_pos() == 12, "abs pos keeps growing");
  expect(cache.k_view(0).dim(1) == 8, "k_view width is occupancy");

  // Sink tokens (first two abs positions) stay at front after slides.
  const float* kd = cache.k_view(0).f32();
  expect(kd != nullptr, "k data");
  expect(kd[0] == 0.f, "sink token 0");
  expect(kd[4] == 1.f, "sink token 1");

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_kv_cache_test: OK\n";
  return EXIT_SUCCESS;
}
