#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/kernels/tensor_view.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace uaii {
namespace runtime {

/// Per-layer K/V cache for autoregressive decode. Layout: [batch, max_seq, dim] f32.
class UAII_API KvCache {
 public:
  KvCache() = default;
  ~KvCache() = default;

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  [[nodiscard]] Error configure(std::int64_t n_layers,
                                std::int64_t batch,
                                std::int64_t max_seq,
                                std::int64_t dim,
                                std::int64_t n_heads) {
    if (n_layers <= 0 || batch <= 0 || max_seq <= 0 || dim <= 0 || n_heads <= 0) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache::configure invalid dims");
    }
    if (dim % n_heads != 0) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache dim not divisible by n_heads");
    }
    n_layers_ = n_layers;
    batch_ = batch;
    max_seq_ = max_seq;
    dim_ = dim;
    n_heads_ = n_heads;
    past_len_ = 0;
    pending_seq_ = 0;
    const std::size_t elems =
        static_cast<std::size_t>(batch_) * static_cast<std::size_t>(max_seq_) *
        static_cast<std::size_t>(dim_);
    k_.assign(static_cast<std::size_t>(n_layers_), std::vector<float>(elems, 0.f));
    v_.assign(static_cast<std::size_t>(n_layers_), std::vector<float>(elems, 0.f));
    return Error::success();
  }

  void reset() noexcept {
    past_len_ = 0;
    pending_seq_ = 0;
    for (auto& buf : k_) std::fill(buf.begin(), buf.end(), 0.f);
    for (auto& buf : v_) std::fill(buf.begin(), buf.end(), 0.f);
  }

  [[nodiscard]] std::int64_t current_seq() const noexcept { return past_len_; }
  [[nodiscard]] std::int64_t past_len() const noexcept { return past_len_; }
  [[nodiscard]] std::int64_t max_seq() const noexcept { return max_seq_; }
  [[nodiscard]] std::int64_t n_layers() const noexcept { return n_layers_; }
  [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }
  [[nodiscard]] std::int64_t dim() const noexcept { return dim_; }
  [[nodiscard]] std::int64_t n_heads() const noexcept { return n_heads_; }
  [[nodiscard]] bool configured() const noexcept { return n_layers_ > 0; }

  [[nodiscard]] Error ensure_capacity(std::int64_t seq) const {
    if (!configured()) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache not configured");
    }
    if (seq < 0 || seq > max_seq_) {
      return Error::make(ErrorCode::InvalidArgument,
                         "KvCache capacity exceeded: seq=" + std::to_string(seq) +
                             " max_seq=" + std::to_string(max_seq_));
    }
    return Error::success();
  }

  /// Append new K/V for one layer at the current past_len_. Does not advance past_len_
  /// until commit_step(). k/v: [batch, seq, dim] or [batch, dim] (seq=1).
  [[nodiscard]] Error append_layer(std::int64_t layer,
                                   const kernels::TensorView& k,
                                   const kernels::TensorView& v) {
    if (!configured()) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache not configured");
    }
    if (layer < 0 || layer >= n_layers_) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache layer out of range");
    }
    std::int64_t batch = 0;
    std::int64_t seq = 0;
    std::int64_t dim = 0;
    Error err = parse_kv_shape(k, &batch, &seq, &dim);
    if (!err.ok()) return err;
    std::int64_t vb = 0, vs = 0, vd = 0;
    err = parse_kv_shape(v, &vb, &vs, &vd);
    if (!err.ok()) return err;
    if (batch != batch_ || vb != batch_ || dim != dim_ || vd != dim_ || seq != vs) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache append shape mismatch");
    }
    err = ensure_capacity(past_len_ + seq);
    if (!err.ok()) return err;

    const float* ks = k.f32();
    const float* vs_ptr = v.f32();
    float* kd = k_[static_cast<std::size_t>(layer)].data();
    float* vd_ptr = v_[static_cast<std::size_t>(layer)].data();
    for (std::int64_t b = 0; b < batch_; ++b) {
      for (std::int64_t s = 0; s < seq; ++s) {
        const std::size_t src =
            static_cast<std::size_t>((b * seq + s) * dim_);
        const std::size_t dst = static_cast<std::size_t>(
            (b * max_seq_ + past_len_ + s) * dim_);
        std::memcpy(kd + dst, ks + src, static_cast<std::size_t>(dim_) * sizeof(float));
        std::memcpy(vd_ptr + dst, vs_ptr + src,
                    static_cast<std::size_t>(dim_) * sizeof(float));
      }
    }
    pending_seq_ = seq;
    return Error::success();
  }

  /// Advance past_len_ by the seq appended this step (call once after a full forward).
  void commit_step() noexcept {
    past_len_ += pending_seq_;
    pending_seq_ = 0;
  }

  /// View of committed past K/V: [batch, past_len, dim]. Empty (null data) if past_len==0.
  [[nodiscard]] kernels::TensorView k_view(std::int64_t layer) const {
    return make_past_view(layer, /*is_k=*/true);
  }
  [[nodiscard]] kernels::TensorView v_view(std::int64_t layer) const {
    return make_past_view(layer, /*is_k=*/false);
  }

 private:
  static Error parse_kv_shape(const kernels::TensorView& t,
                              std::int64_t* batch,
                              std::int64_t* seq,
                              std::int64_t* dim) {
    if (batch == nullptr || seq == nullptr || dim == nullptr) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache parse null");
    }
    if (t.dtype != DType::F32 || t.data == nullptr) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache expects f32 views");
    }
    if (t.rank == 2) {
      *batch = t.dim(0);
      *seq = 1;
      *dim = t.dim(1);
      return Error::success();
    }
    if (t.rank == 3) {
      *batch = t.dim(0);
      *seq = t.dim(1);
      *dim = t.dim(2);
      return Error::success();
    }
    return Error::make(ErrorCode::InvalidArgument, "KvCache expects rank 2 or 3");
  }

  kernels::TensorView make_past_view(std::int64_t layer, bool is_k) const {
    kernels::TensorView v;
    if (!configured() || layer < 0 || layer >= n_layers_ || past_len_ <= 0) {
      return v;
    }
    view_shape_[0] = batch_;
    view_shape_[1] = past_len_;
    view_shape_[2] = dim_;
    v.dtype = DType::F32;
    v.shape = view_shape_;
    v.rank = 3;
    v.data = const_cast<float*>(
        (is_k ? k_ : v_)[static_cast<std::size_t>(layer)].data());
    v.nbytes = static_cast<std::size_t>(batch_) * static_cast<std::size_t>(past_len_) *
               static_cast<std::size_t>(dim_) * sizeof(float);
    return v;
  }

  std::int64_t n_layers_ = 0;
  std::int64_t batch_ = 0;
  std::int64_t max_seq_ = 0;
  std::int64_t dim_ = 0;
  std::int64_t n_heads_ = 0;
  std::int64_t past_len_ = 0;
  std::int64_t pending_seq_ = 0;
  std::vector<std::vector<float>> k_;
  std::vector<std::vector<float>> v_;
  mutable std::int64_t view_shape_[3]{};
};

}  // namespace runtime

namespace kernels {

/// Thread-local active KV cache binding for Attention kernels (set by Session).
UAII_API void set_active_kv_cache(runtime::KvCache* cache) noexcept;
[[nodiscard]] UAII_API runtime::KvCache* active_kv_cache() noexcept;

}  // namespace kernels
}  // namespace uaii
