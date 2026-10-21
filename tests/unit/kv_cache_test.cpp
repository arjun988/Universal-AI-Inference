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

  // MLA: K and V widths differ. Both views must stay valid together.
  {
    KvCache mla;
    expect(mla.configure(1, 1, 4, /*k_dim=*/6, /*v_dim=*/4, /*n_heads=*/2, /*sink=*/0).ok(),
           "mla configure");
    std::vector<float> mk = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> mv = {7.f, 8.f, 9.f, 10.f};
    static std::int64_t ksh[2] = {1, 6};
    static std::int64_t vsh[2] = {1, 4};
    uaii::kernels::TensorView mkv;
    mkv.dtype = uaii::DType::F32;
    mkv.shape = ksh;
    mkv.rank = 2;
    mkv.data = mk.data();
    mkv.nbytes = mk.size() * sizeof(float);
    uaii::kernels::TensorView mvv = mkv;
    mvv.shape = vsh;
    mvv.data = mv.data();
    mvv.nbytes = mv.size() * sizeof(float);
    expect(mla.append_layer(0, mkv, mvv).ok(), "mla append");
    mla.commit_step();
    const auto pk = mla.k_view(0);
    const auto pv = mla.v_view(0);
    expect(pk.dim(2) == 6, "past K keeps k_dim after V view");
    expect(pv.dim(2) == 4, "past V keeps v_dim");
    expect(pk.f32() != nullptr && pk.f32()[0] == 1.f && pk.f32()[5] == 6.f, "past K values");
    expect(pv.f32() != nullptr && pv.f32()[0] == 7.f && pv.f32()[3] == 10.f, "past V values");
  }

  // batch>1 storage is strided by max_seq. The view must be packed [batch, past, dim].
  {
    KvCache batched;
    expect(batched.configure(1, /*batch=*/2, /*max_seq=*/4, 2, 2, 1, 0).ok(), "batch configure");
    std::vector<float> bk = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> bv = {5.f, 6.f, 7.f, 8.f};
    static std::int64_t sh[2] = {2, 2};
    uaii::kernels::TensorView bkv;
    bkv.dtype = uaii::DType::F32;
    bkv.shape = sh;
    bkv.rank = 2;
    bkv.data = bk.data();
    bkv.nbytes = bk.size() * sizeof(float);
    uaii::kernels::TensorView bvv = bkv;
    bvv.data = bv.data();
    bvv.nbytes = bv.size() * sizeof(float);
    expect(batched.append_layer(0, bkv, bvv).ok(), "batch append");
    batched.commit_step();
    const auto pk = batched.k_view(0);
    expect(pk.dim(0) == 2 && pk.dim(1) == 1 && pk.dim(2) == 2, "packed batch shape");
    const float* p = pk.f32();
    expect(p != nullptr && p[0] == 1.f && p[1] == 2.f && p[2] == 3.f && p[3] == 4.f,
           "batch 1 is not read from batch 0 padding");
  }

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_kv_cache_test: OK\n";
  return EXIT_SUCCESS;
}
