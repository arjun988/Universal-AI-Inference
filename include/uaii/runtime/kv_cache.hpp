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

/// Per-layer K/V cache for autoregressive decode.
/// Layout: K [batch, max_seq, k_dim], V [batch, max_seq, v_dim] f32.
/// Dense MHA/GQA uses k_dim == v_dim; MLA expand-then-attend may differ.
class UAII_API KvCache {
 public:
  KvCache() = default;
  ~KvCache() = default;

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  /// Configure with shared K/V width (dense / GQA).
  [[nodiscard]] Error configure(std::int64_t n_layers,
                                std::int64_t batch,
                                std::int64_t max_seq,
                                std::int64_t dim,
                                std::int64_t n_heads) {
    return configure(n_layers, batch, max_seq, dim, dim, n_heads);
  }

  /// Configure with possibly different K and V widths (MLA).
  /// `max_seq` is the **resident window** (not the model's 128k).
  /// `sink` keeps the first N tokens (StreamingLLM-style) when sliding.
  [[nodiscard]] Error configure(std::int64_t n_layers,
                                std::int64_t batch,
                                std::int64_t max_seq,
                                std::int64_t k_dim,
                                std::int64_t v_dim,
                                std::int64_t n_heads,
                                std::int64_t sink = 4) {
    if (n_layers <= 0 || batch <= 0 || max_seq <= 0 || k_dim <= 0 || v_dim <= 0 ||
        n_heads <= 0) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache::configure invalid dims");
    }
    if (k_dim % n_heads != 0) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache k_dim not divisible by n_heads");
    }
    n_layers_ = n_layers;
    batch_ = batch;
    max_seq_ = max_seq;
    k_dim_ = k_dim;
    v_dim_ = v_dim;
    dim_ = k_dim;  // legacy alias (K width)
    n_heads_ = n_heads;
    sink_ = sink < 0 ? 0 : sink;
    if (sink_ >= max_seq_) sink_ = 0;
    past_len_ = 0;
    pending_seq_ = 0;
    abs_pos_ = 0;
    const std::size_t k_elems =
        static_cast<std::size_t>(batch_) * static_cast<std::size_t>(max_seq_) *
        static_cast<std::size_t>(k_dim_);
    const std::size_t v_elems =
        static_cast<std::size_t>(batch_) * static_cast<std::size_t>(max_seq_) *
        static_cast<std::size_t>(v_dim_);
    k_.assign(static_cast<std::size_t>(n_layers_), std::vector<float>(k_elems, 0.f));
    v_.assign(static_cast<std::size_t>(n_layers_), std::vector<float>(v_elems, 0.f));
    return Error::success();
  }

  void reset() noexcept {
    past_len_ = 0;
    pending_seq_ = 0;
    abs_pos_ = 0;
    for (auto& buf : k_) std::fill(buf.begin(), buf.end(), 0.f);
    for (auto& buf : v_) std::fill(buf.begin(), buf.end(), 0.f);
  }

  [[nodiscard]] std::int64_t current_seq() const noexcept { return past_len_; }
  [[nodiscard]] std::int64_t past_len() const noexcept { return past_len_; }
  /// Absolute RoPE position of the token currently being encoded (does not rewind on slide).
  [[nodiscard]] std::int64_t rope_pos() const noexcept { return abs_pos_; }
  [[nodiscard]] std::int64_t sink() const noexcept { return sink_; }
  [[nodiscard]] std::int64_t max_seq() const noexcept { return max_seq_; }
  [[nodiscard]] std::int64_t n_layers() const noexcept { return n_layers_; }
  [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }
  [[nodiscard]] std::int64_t dim() const noexcept { return dim_; }
  [[nodiscard]] std::int64_t k_dim() const noexcept { return k_dim_; }
  [[nodiscard]] std::int64_t v_dim() const noexcept { return v_dim_; }
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
    std::int64_t kd = 0;
    Error err = parse_kv_shape(k, &batch, &seq, &kd);
    if (!err.ok()) return err;
    std::int64_t vb = 0, vs = 0, vd = 0;
    err = parse_kv_shape(v, &vb, &vs, &vd);
    if (!err.ok()) return err;
    if (batch != batch_ || vb != batch_ || kd != k_dim_ || vd != v_dim_ || seq != vs) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache append shape mismatch");
    }
    if (seq > max_seq_) {
      return Error::make(ErrorCode::InvalidArgument, "KvCache append seq exceeds window");
    }
    compact_to_fit(seq);
    err = ensure_capacity(past_len_ + seq);
    if (!err.ok()) return err;

    const float* ks = k.f32();
    const float* vs_ptr = v.f32();
    float* kdst = k_[static_cast<std::size_t>(layer)].data();
    float* vdst = v_[static_cast<std::size_t>(layer)].data();
    for (std::int64_t b = 0; b < batch_; ++b) {
      for (std::int64_t s = 0; s < seq; ++s) {
        const std::size_t src_k =
            static_cast<std::size_t>((b * seq + s) * k_dim_);
        const std::size_t dst_k = static_cast<std::size_t>(
            (b * max_seq_ + past_len_ + s) * k_dim_);
        const std::size_t src_v =
            static_cast<std::size_t>((b * seq + s) * v_dim_);
        const std::size_t dst_v = static_cast<std::size_t>(
            (b * max_seq_ + past_len_ + s) * v_dim_);
        std::memcpy(kdst + dst_k, ks + src_k,
                    static_cast<std::size_t>(k_dim_) * sizeof(float));
        std::memcpy(vdst + dst_v, vs_ptr + src_v,
                    static_cast<std::size_t>(v_dim_) * sizeof(float));
      }
    }
    pending_seq_ = seq;
    return Error::success();
  }

  /// Advance occupancy and the absolute RoPE counter (call once after a full forward).
  void commit_step() noexcept {
    abs_pos_ += pending_seq_;
    past_len_ += pending_seq_;
    pending_seq_ = 0;
  }

  /// View of committed past K/V: contiguous [batch, past_len, k_dim/v_dim].
  /// Empty if past_len==0. K and V views may be held at the same time; each side
  /// keeps its own shape. A later call for the same side invalidates that side's view.
  [[nodiscard]] kernels::TensorView k_view(std::int64_t layer) const {
    return make_past_view(layer, /*is_k=*/true);
  }
  [[nodiscard]] kernels::TensorView v_view(std::int64_t layer) const {
    return make_past_view(layer, /*is_k=*/false);
  }

 private:
  void compact_to_fit(std::int64_t incoming) {
    const std::int64_t need = past_len_ + incoming;
    if (need <= max_seq_) return;
    std::int64_t drop = need - max_seq_;
    const std::int64_t keep_sink = std::min(sink_, past_len_);
    const std::int64_t movable = past_len_ - keep_sink;
    if (drop > movable) drop = movable;
    if (drop <= 0) return;
    const std::int64_t tail = past_len_ - keep_sink - drop;
    for (std::int64_t layer = 0; layer < n_layers_; ++layer) {
      float* kdst = k_[static_cast<std::size_t>(layer)].data();
      float* vdst = v_[static_cast<std::size_t>(layer)].data();
      for (std::int64_t b = 0; b < batch_; ++b) {
        if (tail > 0) {
          const std::size_t k_dst = static_cast<std::size_t>(
              (b * max_seq_ + keep_sink) * k_dim_);
          const std::size_t k_src = static_cast<std::size_t>(
              (b * max_seq_ + keep_sink + drop) * k_dim_);
          std::memmove(kdst + k_dst, kdst + k_src,
                       static_cast<std::size_t>(tail * k_dim_) * sizeof(float));
          const std::size_t v_dst = static_cast<std::size_t>(
              (b * max_seq_ + keep_sink) * v_dim_);
          const std::size_t v_src = static_cast<std::size_t>(
              (b * max_seq_ + keep_sink + drop) * v_dim_);
          std::memmove(vdst + v_dst, vdst + v_src,
                       static_cast<std::size_t>(tail * v_dim_) * sizeof(float));
        }
      }
    }
    past_len_ -= drop;
  }

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
    const std::int64_t d = is_k ? k_dim_ : v_dim_;
    auto& shape = is_k ? k_view_shape_ : v_view_shape_;
    shape[0] = batch_;
    shape[1] = past_len_;
    shape[2] = d;
    v.dtype = DType::F32;
    v.shape = shape;
    v.rank = 3;
    const float* src = (is_k ? k_ : v_)[static_cast<std::size_t>(layer)].data();
    const std::size_t n =
        static_cast<std::size_t>(batch_) * static_cast<std::size_t>(past_len_) *
        static_cast<std::size_t>(d);
    // Storage is [batch, max_seq, dim]. That prefix is contiguous only for batch 1.
    // Attention indexes a packed [batch, past_len, dim] view, so pack when the
    // resident window is wider than the occupancy.
    if (batch_ == 1 || past_len_ == max_seq_) {
      v.data = const_cast<float*>(src);
      v.nbytes = n * sizeof(float);
      return v;
    }
    auto& packed = is_k ? k_packed_ : v_packed_;
    packed.resize(n);
    for (std::int64_t b = 0; b < batch_; ++b) {
      std::memcpy(packed.data() + static_cast<std::size_t>(b * past_len_ * d),
                  src + static_cast<std::size_t>(b * max_seq_ * d),
                  static_cast<std::size_t>(past_len_ * d) * sizeof(float));
    }
    v.data = packed.data();
    v.nbytes = n * sizeof(float);
    return v;
  }

  std::int64_t n_layers_ = 0;
  std::int64_t batch_ = 0;
  std::int64_t max_seq_ = 0;
  std::int64_t dim_ = 0;
  std::int64_t k_dim_ = 0;
  std::int64_t v_dim_ = 0;
  std::int64_t n_heads_ = 0;
  std::int64_t past_len_ = 0;
  std::int64_t pending_seq_ = 0;
  std::int64_t abs_pos_ = 0;
  std::int64_t sink_ = 4;
  std::vector<std::vector<float>> k_;
  std::vector<std::vector<float>> v_;
  mutable std::int64_t k_view_shape_[3]{};
  mutable std::int64_t v_view_shape_[3]{};
  mutable std::vector<float> k_packed_;
  mutable std::vector<float> v_packed_;
};

}  // namespace runtime

namespace kernels {

/// Thread-local active KV cache binding for Attention kernels (set by Session).
UAII_API void set_active_kv_cache(runtime::KvCache* cache) noexcept;
[[nodiscard]] UAII_API runtime::KvCache* active_kv_cache() noexcept;

}  // namespace kernels
}  // namespace uaii
