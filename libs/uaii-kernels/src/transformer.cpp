#include "uaii/kernels/kernels.hpp"
#include "uaii/kernels/view_util.hpp"
#include "uaii/quant/formats.hpp"
#include "uaii/quant/gguf_dequant.hpp"
#include "uaii/runtime/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace uaii {
namespace kernels {
namespace {

thread_local runtime::KvCache* g_active_kv_cache = nullptr;

struct AttnShape {
  std::int64_t batch = 0;
  std::int64_t seq = 0;
  std::int64_t dim = 0;
  bool rank2 = false;
};

Error parse_attn_shape(const TensorView& t, AttnShape* out, const char* name) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "attn shape null");
  }
  if (t.dtype != DType::F32 || t.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, std::string(name) + " f32 required");
  }
  if (t.rank == 2) {
    out->batch = t.dim(0);
    out->seq = 1;
    out->dim = t.dim(1);
    out->rank2 = true;
    return Error::success();
  }
  if (t.rank == 3) {
    out->batch = t.dim(0);
    out->seq = t.dim(1);
    out->dim = t.dim(2);
    out->rank2 = false;
    return Error::success();
  }
  return Error::make(ErrorCode::InvalidArgument,
                     std::string(name) + " expects [batch,seq,dim] or [batch,dim]");
}

}  // namespace

void set_active_kv_cache(runtime::KvCache* cache) noexcept {
  g_active_kv_cache = cache;
}

runtime::KvCache* active_kv_cache() noexcept {
  return g_active_kv_cache;
}

Error embedding_f32(const TensorView& tokens,
                    const TensorView& weight,
                    TensorView* out) {
  if (out == nullptr || tokens.data == nullptr || weight.data == nullptr ||
      out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "embedding null");
  }
  if (tokens.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "embedding requires f32 tokens/out");
  }
  if (weight.rank != 2 || tokens.rank != 2) {
    return Error::make(ErrorCode::InvalidArgument, "embedding rank (tokens/weight)");
  }
  Error err = check_view_bytes(tokens, "tokens");
  if (!err.ok()) return err;

  const std::int64_t vocab =
      weight.quant_rows > 0 ? weight.quant_rows : weight.dim(0);
  const std::int64_t dim =
      weight.quant_cols > 0 ? weight.quant_cols : weight.dim(1);
  const std::int64_t batch = tokens.dim(0);
  const std::int64_t seq = tokens.dim(1);
  // out: [batch, seq, dim] or flattened [batch*seq, dim]
  const bool out_3d = out->rank == 3;
  const bool out_2d = out->rank == 2;
  if (!out_3d && !out_2d) {
    return Error::make(ErrorCode::InvalidArgument, "embedding out rank must be 2 or 3");
  }
  if (out_3d) {
    if (out->dim(0) != batch || out->dim(1) != seq || out->dim(2) != dim) {
      return Error::make(ErrorCode::InvalidArgument, "embedding out shape [B,S,D]");
    }
  } else if (seq == 1) {
    if (out->dim(0) != batch || out->dim(1) != dim) {
      return Error::make(ErrorCode::InvalidArgument, "embedding out shape [B,D]");
    }
  } else {
    if (out->dim(0) != batch * seq || out->dim(1) != dim) {
      return Error::make(ErrorCode::InvalidArgument, "embedding out shape [B*S,D]");
    }
  }
  err = check_view_bytes(*out, "embedding_out");
  if (!err.ok()) return err;

  const float* ids = tokens.f32();
  float* y = out->f32();
  const bool quantized = quant::is_gguf_block_quant(weight.quant_format);
  if (!quantized) {
    if (weight.dtype != DType::F32) {
      return Error::make(ErrorCode::InvalidArgument, "embedding weight requires f32");
    }
    err = check_view_bytes(weight, "weight");
    if (!err.ok()) return err;
  }

  const float* w = quantized ? nullptr : weight.f32();
  const auto* packed = quantized ? static_cast<const std::uint8_t*>(weight.data) : nullptr;
  const std::size_t row_stride =
      quantized ? quant::packed_nbytes(weight.quant_format, static_cast<std::size_t>(dim)) : 0;

  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t s = 0; s < seq; ++s) {
      const int id = static_cast<int>(ids[b * seq + s]);
      if (id < 0 || id >= static_cast<int>(vocab)) {
        return Error::make(ErrorCode::InvalidArgument,
                           "embedding token id out of range: " + std::to_string(id) +
                               " vocab=" + std::to_string(vocab));
      }
      const std::int64_t row = b * seq + s;
      float* dst = y + row * dim;
      if (quantized) {
        err = quant::dequant_gguf_row(weight.quant_format, packed + static_cast<std::size_t>(id) * row_stride,
                                      dim, dst);
        if (!err.ok()) return err;
      } else {
        std::memcpy(dst, w + static_cast<std::int64_t>(id) * dim,
                    static_cast<std::size_t>(dim) * sizeof(float));
      }
    }
  }
  return Error::success();
}

float rope_scaled_inv_freq(std::int64_t pair_index, std::int64_t n_pairs, float theta,
                           const char* scale_type, float scale_factor, float low_freq_factor,
                           float high_freq_factor, float old_context_len, float seq_len,
                           const float* short_factor, const float* long_factor, int n_factors) {
  if (n_pairs <= 0 || pair_index < 0) return 0.f;
  const float dim = static_cast<float>(n_pairs * 2);
  const float i = static_cast<float>(pair_index);
  const float base_theta = theta > 0.f ? theta : 10000.f;
  float inv = 1.f / std::pow(base_theta, (2.f * i) / dim);
  const std::string kind = scale_type ? scale_type : "";
  if (kind.empty() || kind == "none" || kind == "default") {
    return inv;
  }
  const float factor = scale_factor > 0.f ? scale_factor : 1.f;
  const float old_ctx = old_context_len > 0.f ? old_context_len : 8192.f;
  const float seq = seq_len > 0.f ? seq_len : old_ctx;
  if (kind == "linear") {
    return inv / factor;
  }
  if (kind == "dynamic") {
    // Hugging Face DynamicNTKScalingRotaryEmbedding: scale base when seq > orig ctx.
    if (dim > 2.f && seq > old_ctx) {
      const float ntk = factor * (seq / old_ctx) - (factor - 1.f);
      if (ntk > 0.f) {
        const float scaled_theta = base_theta * std::pow(ntk, dim / (dim - 2.f));
        inv = 1.f / std::pow(scaled_theta, (2.f * i) / dim);
      }
    }
    return inv;
  }
  if (kind == "longrope" || kind == "su") {
    const float* fac = (seq > old_ctx) ? long_factor : short_factor;
    if (fac == nullptr) fac = short_factor;
    if (fac != nullptr && pair_index < n_factors) {
      const float f = fac[pair_index];
      if (f > 0.f) inv /= f;
    }
    return inv;
  }
  if (kind == "llama3") {
    const float low_f = low_freq_factor > 0.f ? low_freq_factor : 1.f;
    const float high_f = high_freq_factor > 0.f ? high_freq_factor : 4.f;
    constexpr float kPi = 3.14159265358979323846f;
    const float wavelen = 2.f * kPi / inv;
    const float low_w = old_ctx / low_f;
    const float high_w = old_ctx / high_f;
    if (wavelen < high_w) return inv;
    if (wavelen > low_w) return inv / factor;
    const float denom = high_f - low_f;
    const float smooth = denom != 0.f ? (old_ctx / wavelen - low_f) / denom : 0.f;
    return (1.f - smooth) * (inv / factor) + smooth * inv;
  }
  if (kind == "yarn") {
    // Hugging Face YaRN: interpolation vs extrapolation ramp (beta_fast=32, beta_slow=1).
    constexpr float kPi = 3.14159265358979323846f;
    const float yarn_ctx = old_context_len > 0.f ? old_context_len : 2048.f;
    const float inv_extrap = inv;
    const float inv_interp = inv / factor;
    auto correction_dim = [&](float num_rotations) -> float {
      if (!(base_theta > 1.f) || !(yarn_ctx > 0.f) || num_rotations <= 0.f) return 0.f;
      return (dim * std::log(yarn_ctx / (num_rotations * 2.f * kPi))) /
             (2.f * std::log(base_theta));
    };
    float low = std::floor(correction_dim(32.f));
    float high = std::ceil(correction_dim(1.f));
    if (low < 0.f) low = 0.f;
    if (high > dim - 1.f) high = dim - 1.f;
    if (low == high) high += 0.001f;
    float ramp = (i - low) / (high - low);
    if (ramp < 0.f) ramp = 0.f;
    if (ramp > 1.f) ramp = 1.f;
    // ramp 0 (high frequency) stays on the original wavelength.
    // ramp 1 (low frequency) uses the interpolated 1/factor wavelength.
    // HF: inv = interp * ramp + extrap * (1 - ramp).
    return inv_interp * ramp + inv_extrap * (1.f - ramp);
  }
  return inv;
}

Error rope_f32(const TensorView& in,
               const TensorView* positions,
               TensorView* out,
               float theta,
               std::int64_t head_dim,
               int mode,
               std::int64_t rotary_dim,
               const char* scale_type,
               float scale_factor,
               float low_freq_factor,
               float high_freq_factor,
               float old_context_len,
               float attn_factor,
               const float* short_factor,
               const float* long_factor,
               int n_factors) {
  if (out == nullptr || in.data == nullptr || out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "rope null");
  }
  if (in.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "rope f32");
  }
  if (in.numel() != out->numel() || in.rank < 2) {
    return Error::make(ErrorCode::InvalidArgument, "rope shape");
  }
  const std::int64_t dim = in.dim(in.rank - 1);
  if (dim % 2 != 0) {
    return Error::make(ErrorCode::InvalidArgument, "rope dim must be even");
  }
  // Apply RoPE per head when head_dim is set (Qwen/Llama MHA+GQA).
  std::int64_t rope_chunk = head_dim > 0 ? head_dim : dim;
  if (rope_chunk <= 0 || dim % rope_chunk != 0 || rope_chunk % 2 != 0) {
    return Error::make(ErrorCode::InvalidArgument, "rope head_dim invalid");
  }
  // DeepSeek MLA: rotate only trailing rotary_dim of each head; copy the prefix.
  std::int64_t rot = rotary_dim > 0 ? rotary_dim : rope_chunk;
  if (rot > rope_chunk || rot <= 0 || rot % 2 != 0) {
    return Error::make(ErrorCode::InvalidArgument, "rope rotary_dim invalid");
  }
  const std::int64_t rot_off = rope_chunk - rot;
  const bool neox = (mode == 2);
  const std::int64_t n_chunks = dim / rope_chunk;
  const std::int64_t half = rot / 2;
  std::size_t rows = in.numel() / static_cast<std::size_t>(dim);
  const float* x = in.f32();
  float* y = out->f32();

  const std::string scale_kind = scale_type ? scale_type : "";
  if (!scale_kind.empty() && scale_kind != "none" && scale_kind != "default" &&
      scale_kind != "linear" && scale_kind != "llama3" && scale_kind != "dynamic" &&
      scale_kind != "yarn" && scale_kind != "longrope" && scale_kind != "su") {
    return Error::make(ErrorCode::NotImplemented,
                       "unsupported rope_scaling.type=" + scale_kind);
  }

  float yarn_mscale = 1.f;
  if (scale_kind == "yarn") {
    const float factor = scale_factor > 0.f ? scale_factor : 1.f;
    const float af = attn_factor > 0.f ? attn_factor : 1.f;
    if (factor > 1.f) {
      yarn_mscale = 0.1f * af * std::log(factor) + 1.f;
    }
  }

  // Default position: KV cache length (seq=1 decode) or row index.
  const std::int64_t base_pos =
      (active_kv_cache() != nullptr) ? active_kv_cache()->rope_pos() : 0;

  for (std::size_t r = 0; r < rows; ++r) {
    float pos = static_cast<float>(base_pos + static_cast<std::int64_t>(r));
    if (positions && positions->data && positions->numel() > r) {
      pos = positions->f32()[r];
    }
    const float* row = x + r * static_cast<std::size_t>(dim);
    float* out_row = y + r * static_cast<std::size_t>(dim);
    for (std::int64_t h = 0; h < n_chunks; ++h) {
      const float* head = row + static_cast<std::size_t>(h * rope_chunk);
      float* out_head = out_row + static_cast<std::size_t>(h * rope_chunk);
      if (rot_off > 0) {
        std::memcpy(out_head, head, static_cast<std::size_t>(rot_off) * sizeof(float));
      }
      const float* rot_in = head + rot_off;
      float* rot_out = out_head + rot_off;
      if (neox) {
        // NeoX / GPT-NeoX: pair (i, i + half) — required by Qwen2/Phi/Falcon.
        for (std::int64_t i = 0; i < half; ++i) {
          const float freq = rope_scaled_inv_freq(
              i, half, theta, scale_type, scale_factor, low_freq_factor, high_freq_factor,
              old_context_len, pos + 1.f, short_factor, long_factor, n_factors);
          const float angle = pos * freq;
          const float c = std::cos(angle) * yarn_mscale;
          const float s = std::sin(angle) * yarn_mscale;
          const float a = rot_in[i];
          const float b = rot_in[i + half];
          rot_out[i] = a * c - b * s;
          rot_out[i + half] = a * s + b * c;
        }
      } else {
        // Interleaved pairs (i, i+1) — llama.cpp NORMAL / GPT-J style.
        for (std::int64_t i = 0; i < rot; i += 2) {
          const float freq = rope_scaled_inv_freq(
              i / 2, half, theta, scale_type, scale_factor, low_freq_factor,
              high_freq_factor, old_context_len, pos + 1.f, short_factor, long_factor,
              n_factors);
          const float angle = pos * freq;
          const float c = std::cos(angle) * yarn_mscale;
          const float s = std::sin(angle) * yarn_mscale;
          const float a = rot_in[i];
          const float b = rot_in[i + 1];
          rot_out[i] = a * c - b * s;
          rot_out[i + 1] = a * s + b * c;
        }
      }
    }
  }
  return Error::success();
}

Error mla_expand_k_f32(const TensorView& k_nope,
                       const TensorView& k_rope,
                       TensorView* out,
                       int num_heads,
                       std::int64_t nope_dim,
                       std::int64_t rope_dim) {
  if (out == nullptr || k_nope.data == nullptr || k_rope.data == nullptr ||
      out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "mla_expand_k null");
  }
  if (k_nope.dtype != DType::F32 || k_rope.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "mla_expand_k f32");
  }
  if (k_nope.rank != 2 || k_rope.rank != 2 || out->rank != 2) {
    return Error::make(ErrorCode::InvalidArgument, "mla_expand_k rank-2");
  }
  if (num_heads <= 0 || nope_dim <= 0 || rope_dim <= 0) {
    return Error::make(ErrorCode::InvalidArgument, "mla_expand_k dims");
  }
  const std::int64_t batch = k_nope.dim(0);
  if (k_rope.dim(0) != batch || out->dim(0) != batch) {
    return Error::make(ErrorCode::InvalidArgument, "mla_expand_k batch");
  }
  if (k_nope.dim(1) != num_heads * nope_dim || k_rope.dim(1) != rope_dim ||
      out->dim(1) != num_heads * (nope_dim + rope_dim)) {
    return Error::make(ErrorCode::InvalidArgument, "mla_expand_k shape");
  }
  const float* nope = k_nope.f32();
  const float* rope = k_rope.f32();
  float* dst = out->f32();
  const std::int64_t out_head = nope_dim + rope_dim;
  for (std::int64_t b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      float* oh = dst + static_cast<std::size_t>(b * num_heads * out_head + h * out_head);
      const float* nh =
          nope + static_cast<std::size_t>(b * num_heads * nope_dim + h * nope_dim);
      const float* rh = rope + static_cast<std::size_t>(b * rope_dim);
      std::memcpy(oh, nh, static_cast<std::size_t>(nope_dim) * sizeof(float));
      std::memcpy(oh + nope_dim, rh, static_cast<std::size_t>(rope_dim) * sizeof(float));
    }
  }
  return Error::success();
}

Error attention_kv_f32(const TensorView& q,
                       const TensorView& k,
                       const TensorView& v,
                       TensorView* out,
                       int num_heads,
                       float scale,
                       bool causal,
                       const TensorView* past_k,
                       const TensorView* past_v,
                       TensorView* present_k,
                       TensorView* present_v,
                       float logit_softcap,
                       std::int64_t sliding_window) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "attention out null");
  }
  AttnShape qs, ks, vs, os;
  Error err = parse_attn_shape(q, &qs, "attn_q");
  if (!err.ok()) return err;
  err = parse_attn_shape(k, &ks, "attn_k");
  if (!err.ok()) return err;
  err = parse_attn_shape(v, &vs, "attn_v");
  if (!err.ok()) return err;
  err = parse_attn_shape(*out, &os, "attn_out");
  if (!err.ok()) return err;

  err = check_view_bytes(q, "attn_q");
  if (!err.ok()) return err;
  err = check_view_bytes(k, "attn_k");
  if (!err.ok()) return err;
  err = check_view_bytes(v, "attn_v");
  if (!err.ok()) return err;
  err = check_view_bytes(*out, "attn_out");
  if (!err.ok()) return err;

  // Q/out: out may match V dim (MLA: v_head_dim may differ from qk head_dim).
  // K/V may differ in width when V head dim != QK head dim.
  if (qs.batch != ks.batch || qs.batch != vs.batch || qs.batch != os.batch ||
      qs.seq != os.seq || ks.seq != vs.seq || qs.seq != ks.seq) {
    return Error::make(ErrorCode::InvalidArgument, "attention shape mismatch");
  }

  const std::int64_t batch = qs.batch;
  const std::int64_t q_seq = qs.seq;
  const std::int64_t new_kv_seq = ks.seq;
  const std::int64_t q_dim = qs.dim;
  const std::int64_t k_dim = ks.dim;
  const std::int64_t v_dim = vs.dim;
  if (num_heads <= 0 || q_dim % num_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "invalid num_heads");
  }
  const std::int64_t head_dim = q_dim / num_heads;
  if (k_dim % head_dim != 0) {
    return Error::make(ErrorCode::InvalidArgument, "k_dim not divisible by head_dim");
  }
  const int num_kv_heads = static_cast<int>(k_dim / head_dim);
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "invalid GQA head grouping");
  }
  if (v_dim % num_kv_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "v_dim not divisible by kv_heads");
  }
  const std::int64_t v_head_dim = v_dim / num_kv_heads;
  // GQA/MHA: out = num_heads * v_head_dim (== q_dim when v_head_dim == head_dim).
  // MLA: v_head_dim may differ from qk head_dim; out still one V vector per Q head.
  const std::int64_t out_dim = static_cast<std::int64_t>(num_heads) * v_head_dim;
  if (os.dim != out_dim) {
    return Error::make(ErrorCode::InvalidArgument, "attention out dim mismatch");
  }
  const int heads_per_kv = num_heads / num_kv_heads;

  std::int64_t past_seq = 0;
  const float* PK = nullptr;
  const float* PV = nullptr;
  if (past_k != nullptr || past_v != nullptr) {
    if (past_k == nullptr || past_v == nullptr || past_k->data == nullptr ||
        past_v->data == nullptr) {
      return Error::make(ErrorCode::InvalidArgument, "attention past_k/past_v incomplete");
    }
    AttnShape pks, pvs;
    err = parse_attn_shape(*past_k, &pks, "past_k");
    if (!err.ok()) return err;
    err = parse_attn_shape(*past_v, &pvs, "past_v");
    if (!err.ok()) return err;
    if (pks.batch != batch || pvs.batch != batch || pks.dim != k_dim || pvs.dim != v_dim ||
        pks.seq != pvs.seq) {
      return Error::make(ErrorCode::InvalidArgument, "attention past shape mismatch");
    }
    err = check_view_bytes(*past_k, "past_k");
    if (!err.ok()) return err;
    err = check_view_bytes(*past_v, "past_v");
    if (!err.ok()) return err;
    past_seq = pks.seq;
    PK = past_k->f32();
    PV = past_v->f32();
  }

  const std::int64_t kv_seq = past_seq + new_kv_seq;
  if (scale <= 0) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  }

  const float* Q = q.f32();
  const float* K = k.f32();
  const float* V = v.f32();
  float* O = out->f32();
  std::vector<float> scores(static_cast<std::size_t>(q_seq * kv_seq));

  auto k_at = [&](std::int64_t b, std::int64_t j, int kv_h, std::int64_t d) -> float {
    if (j < past_seq) {
      return PK[static_cast<std::size_t>(((b * past_seq + j) * k_dim) + kv_h * head_dim + d)];
    }
    const std::int64_t jj = j - past_seq;
    return K[static_cast<std::size_t>(((b * new_kv_seq + jj) * k_dim) + kv_h * head_dim + d)];
  };
  auto v_at = [&](std::int64_t b, std::int64_t j, int kv_h, std::int64_t d) -> float {
    if (j < past_seq) {
      return PV[static_cast<std::size_t>(((b * past_seq + j) * v_dim) + kv_h * v_head_dim + d)];
    }
    const std::int64_t jj = j - past_seq;
    return V[static_cast<std::size_t>(((b * new_kv_seq + jj) * v_dim) + kv_h * v_head_dim + d)];
  };

  for (std::int64_t b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      const int kv_h = h / heads_per_kv;
      for (std::int64_t i = 0; i < q_seq; ++i) {
        const std::int64_t abs_i = past_seq + i;
        float max_v = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = 0; j < kv_seq; ++j) {
          if (causal && j > abs_i) {
            scores[static_cast<std::size_t>(i * kv_seq + j)] = -1e9f;
            continue;
          }
          if (sliding_window > 0 && j + sliding_window <= abs_i) {
            scores[static_cast<std::size_t>(i * kv_seq + j)] = -1e9f;
            continue;
          }
          float dot = 0;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            const std::size_t qi =
                static_cast<std::size_t>(((b * q_seq + i) * q_dim) + h * head_dim + d);
            dot += Q[qi] * k_at(b, j, kv_h, d);
          }
          dot *= scale;
          if (logit_softcap > 0.f) {
            dot = logit_softcap * std::tanh(dot / logit_softcap);
          }
          scores[static_cast<std::size_t>(i * kv_seq + j)] = dot;
          max_v = std::max(max_v, dot);
        }
        float sum = 0;
        for (std::int64_t j = 0; j < kv_seq; ++j) {
          float e = std::exp(scores[static_cast<std::size_t>(i * kv_seq + j)] - max_v);
          scores[static_cast<std::size_t>(i * kv_seq + j)] = e;
          sum += e;
        }
        const float inv = sum > 0 ? 1.0f / sum : 0.0f;
        for (std::int64_t j = 0; j < kv_seq; ++j) {
          scores[static_cast<std::size_t>(i * kv_seq + j)] *= inv;
        }
      }
      for (std::int64_t i = 0; i < q_seq; ++i) {
        for (std::int64_t d = 0; d < v_head_dim; ++d) {
          float acc = 0;
          for (std::int64_t j = 0; j < kv_seq; ++j) {
            acc += scores[static_cast<std::size_t>(i * kv_seq + j)] * v_at(b, j, kv_h, d);
          }
          const std::size_t oi =
              static_cast<std::size_t>(((b * q_seq + i) * out_dim) + h * v_head_dim + d);
          O[oi] = acc;
        }
      }
    }
  }

  // Write present K/V = concat(past, new). Same data pointer as past ⇒ append in-place.
  if (present_k != nullptr || present_v != nullptr) {
    if (present_k == nullptr || present_v == nullptr) {
      return Error::make(ErrorCode::InvalidArgument, "attention present_k/v incomplete");
    }
    const bool inplace =
        past_k != nullptr && past_v != nullptr && present_k->data == past_k->data &&
        present_v->data == past_v->data;
    const std::size_t need_k =
        static_cast<std::size_t>(batch * kv_seq * k_dim) * sizeof(float);
    const std::size_t need_v =
        static_cast<std::size_t>(batch * kv_seq * v_dim) * sizeof(float);
    if (present_k->nbytes < need_k || present_v->nbytes < need_v) {
      return Error::make(ErrorCode::InvalidArgument, "present buffer too small");
    }
    float* pk = present_k->f32();
    float* pv = present_v->f32();
    // Stride for in-place capacity buffers uses max capacity inferred from nbytes.
    const std::int64_t stride_seq_k =
        inplace ? static_cast<std::int64_t>(present_k->nbytes /
                                           (static_cast<std::size_t>(batch * k_dim) *
                                            sizeof(float)))
                : kv_seq;
    const std::int64_t stride_seq_v =
        inplace ? static_cast<std::int64_t>(present_v->nbytes /
                                           (static_cast<std::size_t>(batch * v_dim) *
                                            sizeof(float)))
                : kv_seq;
    if (stride_seq_k < kv_seq || stride_seq_v < kv_seq) {
      return Error::make(ErrorCode::InvalidArgument, "present stride < kv_seq");
    }
    if (!inplace) {
      for (std::int64_t b = 0; b < batch; ++b) {
        if (past_seq > 0) {
          std::memcpy(pk + static_cast<std::size_t>(b * kv_seq * k_dim),
                      PK + static_cast<std::size_t>(b * past_seq * k_dim),
                      static_cast<std::size_t>(past_seq * k_dim) * sizeof(float));
          std::memcpy(pv + static_cast<std::size_t>(b * kv_seq * v_dim),
                      PV + static_cast<std::size_t>(b * past_seq * v_dim),
                      static_cast<std::size_t>(past_seq * v_dim) * sizeof(float));
        }
        std::memcpy(pk + static_cast<std::size_t>((b * kv_seq + past_seq) * k_dim),
                    K + static_cast<std::size_t>(b * new_kv_seq * k_dim),
                    static_cast<std::size_t>(new_kv_seq * k_dim) * sizeof(float));
        std::memcpy(pv + static_cast<std::size_t>((b * kv_seq + past_seq) * v_dim),
                    V + static_cast<std::size_t>(b * new_kv_seq * v_dim),
                    static_cast<std::size_t>(new_kv_seq * v_dim) * sizeof(float));
      }
    } else {
      for (std::int64_t b = 0; b < batch; ++b) {
        for (std::int64_t s = 0; s < new_kv_seq; ++s) {
          const std::size_t src_k =
              static_cast<std::size_t>((b * new_kv_seq + s) * k_dim);
          const std::size_t dst_k =
              static_cast<std::size_t>((b * stride_seq_k + past_seq + s) * k_dim);
          const std::size_t src_v =
              static_cast<std::size_t>((b * new_kv_seq + s) * v_dim);
          const std::size_t dst_v =
              static_cast<std::size_t>((b * stride_seq_v + past_seq + s) * v_dim);
          std::memcpy(pk + dst_k, K + src_k, static_cast<std::size_t>(k_dim) * sizeof(float));
          std::memcpy(pv + dst_v, V + src_v, static_cast<std::size_t>(v_dim) * sizeof(float));
        }
      }
    }
  }
  return Error::success();
}

Error attention_f32(const TensorView& q,
                    const TensorView& k,
                    const TensorView& v,
                    TensorView* out,
                    int num_heads,
                    float scale,
                    bool causal,
                    float logit_softcap,
                    std::int64_t sliding_window) {
  return attention_kv_f32(q, k, v, out, num_heads, scale, causal, nullptr, nullptr,
                          nullptr, nullptr, logit_softcap, sliding_window);
}

Error positions_f32(const TensorView& tokens, TensorView* out) {
  if (out == nullptr || tokens.data == nullptr || out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "positions null");
  }
  if (tokens.dtype != DType::F32 || out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "positions f32");
  }
  if (tokens.numel() != out->numel()) {
    return Error::make(ErrorCode::InvalidArgument, "positions shape");
  }
  const std::int64_t base =
      (active_kv_cache() != nullptr) ? active_kv_cache()->rope_pos() : 0;
  float* y = out->f32();
  for (std::size_t i = 0; i < tokens.numel(); ++i) {
    y[i] = static_cast<float>(base + static_cast<std::int64_t>(i));
  }
  return Error::success();
}

Error moe_router_f32(const TensorView& x,
                     const TensorView& gate_w,
                     TensorView* probs,
                     TensorView* top_expert) {
  if (probs == nullptr || top_expert == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "moe router outs");
  }
  // x [B,D], gate [E,D], probs [B,E], top [B,1]
  if (x.rank != 2 || gate_w.rank != 2) {
    return Error::make(ErrorCode::InvalidArgument, "moe router ranks");
  }
  TensorView logits = *probs;  // reuse probs buffer for logits then softmax
  Error err = matmul_f32(x, gate_w, &logits, false, true);
  if (!err.ok()) return err;
  err = softmax_f32(logits, probs, -1);
  if (!err.ok()) return err;

  const std::int64_t B = x.dim(0);
  const std::int64_t E = gate_w.dim(0);
  if (top_expert->rank != 2 || top_expert->dim(0) != B || top_expert->dim(1) != 1) {
    return Error::make(ErrorCode::InvalidArgument, "top_expert shape");
  }
  for (std::int64_t b = 0; b < B; ++b) {
    int best = 0;
    float best_p = probs->f32()[static_cast<std::size_t>(b * E)];
    for (std::int64_t e = 1; e < E; ++e) {
      float p = probs->f32()[static_cast<std::size_t>(b * E + e)];
      if (p > best_p) {
        best_p = p;
        best = static_cast<int>(e);
      }
    }
    top_expert->f32()[static_cast<std::size_t>(b)] = static_cast<float>(best);
  }
  return Error::success();
}

Error moe_experts_f32(const TensorView& x,
                      const TensorView& experts_w,
                      const TensorView& top_expert,
                      TensorView* out,
                      int num_experts) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "moe experts out");
  }
  if (x.rank != 2 || out->rank != 2 || experts_w.rank != 3) {
    return Error::make(ErrorCode::InvalidArgument,
                       "moe experts expects x[B,D], W[E,D,D], out[B,D]");
  }
  const std::int64_t B = x.dim(0);
  const std::int64_t D = x.dim(1);
  if (experts_w.dim(0) != num_experts || experts_w.dim(1) != D ||
      experts_w.dim(2) != D) {
    return Error::make(ErrorCode::InvalidArgument, "experts_w shape");
  }
  const float* X = x.f32();
  const float* W = experts_w.f32();
  const float* Eid = top_expert.f32();
  float* Y = out->f32();
  for (std::int64_t b = 0; b < B; ++b) {
    int e = static_cast<int>(Eid[static_cast<std::size_t>(b)]);
    if (e < 0 || e >= num_experts) e = 0;
    const float* We = W + static_cast<std::size_t>(e) * static_cast<std::size_t>(D * D);
    for (std::int64_t o = 0; o < D; ++o) {
      float sum = 0;
      for (std::int64_t i = 0; i < D; ++i) {
        // out = x @ W[e].T  if W stored [D_out, D_in] as row-major W[o,i]
        sum += X[static_cast<std::size_t>(b * D + i)] *
               We[static_cast<std::size_t>(o * D + i)];
      }
      Y[static_cast<std::size_t>(b * D + o)] = sum;
    }
  }
  return Error::success();
}

Error moe_experts_swiglu_f32(const TensorView& x,
                             const TensorView& gate_exps,
                             const TensorView& up_exps,
                             const TensorView& down_exps,
                             const TensorView& probs,
                             TensorView* out,
                             int num_experts,
                             int top_k) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "moe swiglu out");
  }
  if (x.rank != 2 || out->rank != 2 || gate_exps.rank != 3 || up_exps.rank != 3 ||
      down_exps.rank != 3 || probs.rank != 2) {
    return Error::make(ErrorCode::InvalidArgument,
                       "moe swiglu expects x[B,D], gate/up[E,I,D], down[E,D,I], probs[B,E]");
  }
  const std::int64_t B = x.dim(0);
  const std::int64_t D = x.dim(1);
  const std::int64_t E = gate_exps.dim(0);
  const std::int64_t I = gate_exps.dim(1);
  if (E != num_experts || up_exps.dim(0) != E || up_exps.dim(1) != I || up_exps.dim(2) != D ||
      down_exps.dim(0) != E || down_exps.dim(1) != D || down_exps.dim(2) != I ||
      probs.dim(0) != B || probs.dim(1) != E || out->dim(0) != B || out->dim(1) != D ||
      gate_exps.dim(2) != D) {
    return Error::make(ErrorCode::InvalidArgument, "moe swiglu shape mismatch");
  }
  int k = top_k > 0 ? top_k : 1;
  if (k > static_cast<int>(E)) k = static_cast<int>(E);

  const float* X = x.f32();
  const float* G = gate_exps.f32();
  const float* U = up_exps.f32();
  const float* Dn = down_exps.f32();
  const float* P = probs.f32();
  float* Y = out->f32();

  std::vector<float> gate_act(static_cast<std::size_t>(I));
  std::vector<float> up_act(static_cast<std::size_t>(I));
  std::vector<float> hid(static_cast<std::size_t>(I));
  std::vector<float> expert_out(static_cast<std::size_t>(D));
  std::vector<std::pair<float, int>> ranked(static_cast<std::size_t>(E));

  for (std::int64_t b = 0; b < B; ++b) {
    for (std::int64_t e = 0; e < E; ++e) {
      ranked[static_cast<std::size_t>(e)] = {
          P[static_cast<std::size_t>(b * E + e)], static_cast<int>(e)};
    }
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                      [](const auto& a, const auto& b2) { return a.first > b2.first; });
    float mass = 0.0f;
    for (int t = 0; t < k; ++t) mass += ranked[static_cast<std::size_t>(t)].first;
    if (!(mass > 0.0f)) mass = 1.0f;

    for (std::int64_t o = 0; o < D; ++o) Y[static_cast<std::size_t>(b * D + o)] = 0.0f;

    const float* xb = X + static_cast<std::size_t>(b * D);
    for (int t = 0; t < k; ++t) {
      const int e = ranked[static_cast<std::size_t>(t)].second;
      const float w = ranked[static_cast<std::size_t>(t)].first / mass;
      const float* Ge = G + static_cast<std::size_t>(e) * static_cast<std::size_t>(I * D);
      const float* Ue = U + static_cast<std::size_t>(e) * static_cast<std::size_t>(I * D);
      const float* De = Dn + static_cast<std::size_t>(e) * static_cast<std::size_t>(D * I);

      for (std::int64_t i = 0; i < I; ++i) {
        float gsum = 0.0f;
        float usum = 0.0f;
        for (std::int64_t d = 0; d < D; ++d) {
          const float xv = xb[static_cast<std::size_t>(d)];
          gsum += xv * Ge[static_cast<std::size_t>(i * D + d)];
          usum += xv * Ue[static_cast<std::size_t>(i * D + d)];
        }
        // SiLU
        const float sig = 1.0f / (1.0f + std::exp(-gsum));
        gate_act[static_cast<std::size_t>(i)] = gsum * sig;
        up_act[static_cast<std::size_t>(i)] = usum;
        hid[static_cast<std::size_t>(i)] =
            gate_act[static_cast<std::size_t>(i)] * up_act[static_cast<std::size_t>(i)];
      }
      for (std::int64_t o = 0; o < D; ++o) {
        float sum = 0.0f;
        for (std::int64_t i = 0; i < I; ++i) {
          sum += hid[static_cast<std::size_t>(i)] *
                 De[static_cast<std::size_t>(o * I + i)];
        }
        expert_out[static_cast<std::size_t>(o)] = sum;
        Y[static_cast<std::size_t>(b * D + o)] += w * sum;
      }
    }
  }
  return Error::success();
}

Error moe_experts_swiglu_listed_f32(const TensorView& x,
                                    const TensorView& probs,
                                    const TensorView* expert_triples,
                                    int num_experts,
                                    int top_k,
                                    TensorView* out) {
  if (out == nullptr || expert_triples == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "moe listed null");
  }
  if (x.rank != 2 || out->rank != 2 || probs.rank != 2 || num_experts <= 0) {
    return Error::make(ErrorCode::InvalidArgument, "moe listed rank");
  }
  const std::int64_t B = x.dim(0);
  const std::int64_t D = x.dim(1);
  const std::int64_t E = num_experts;
  if (probs.dim(0) != B || probs.dim(1) != E || out->dim(0) != B || out->dim(1) != D) {
    return Error::make(ErrorCode::InvalidArgument, "moe listed shape mismatch");
  }
  const TensorView& g0 = expert_triples[0];
  std::int64_t I = 0;
  for (int e = 0; e < static_cast<int>(E); ++e) {
    const TensorView& Ge = expert_triples[e * 3 + 0];
    if (Ge.data != nullptr && Ge.rank == 2) {
      I = Ge.dim(0);
      break;
    }
  }
  if (I <= 0 && g0.rank == 2) I = g0.dim(0);
  if (I <= 0) {
    return Error::make(ErrorCode::InvalidArgument, "moe listed expert width unknown");
  }
  int k = top_k > 0 ? top_k : 1;
  if (k > static_cast<int>(E)) k = static_cast<int>(E);

  for (int e = 0; e < static_cast<int>(E); ++e) {
    const TensorView& Ge = expert_triples[e * 3 + 0];
    const TensorView& Ue = expert_triples[e * 3 + 1];
    const TensorView& De = expert_triples[e * 3 + 2];
    if (Ge.data == nullptr && Ue.data == nullptr && De.data == nullptr) continue;
    if (Ge.rank != 2 || Ue.rank != 2 || De.rank != 2 || Ge.data == nullptr ||
        Ue.data == nullptr || De.data == nullptr || Ge.dim(0) != I || Ge.dim(1) != D ||
        Ue.dim(0) != I || Ue.dim(1) != D || De.dim(0) != D || De.dim(1) != I) {
      return Error::make(ErrorCode::InvalidArgument,
                         "moe listed expert shape expected gate/up[I,D] down[D,I]");
    }
  }

  const float* X = x.f32();
  const float* P = probs.f32();
  float* Y = out->f32();
  std::vector<float> hid(static_cast<std::size_t>(I));
  std::vector<std::pair<float, int>> ranked(static_cast<std::size_t>(E));

  for (std::int64_t b = 0; b < B; ++b) {
    for (std::int64_t e = 0; e < E; ++e) {
      ranked[static_cast<std::size_t>(e)] = {
          P[static_cast<std::size_t>(b * E + e)], static_cast<int>(e)};
    }
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                      [](const auto& a, const auto& b2) { return a.first > b2.first; });
    float mass = 0.0f;
    for (int t = 0; t < k; ++t) mass += ranked[static_cast<std::size_t>(t)].first;
    if (!(mass > 0.0f)) mass = 1.0f;
    for (std::int64_t o = 0; o < D; ++o) Y[static_cast<std::size_t>(b * D + o)] = 0.0f;

    const float* xb = X + static_cast<std::size_t>(b * D);
    for (int t = 0; t < k; ++t) {
      const int e = ranked[static_cast<std::size_t>(t)].second;
      const float w = ranked[static_cast<std::size_t>(t)].first / mass;
      if (expert_triples[e * 3 + 0].data == nullptr ||
          expert_triples[e * 3 + 1].data == nullptr ||
          expert_triples[e * 3 + 2].data == nullptr) {
        return Error::make(ErrorCode::NotFound,
                           "MoE expert " + std::to_string(e) + " was not paged in");
      }
      const float* Ge = expert_triples[e * 3 + 0].f32();
      const float* Ue = expert_triples[e * 3 + 1].f32();
      const float* De = expert_triples[e * 3 + 2].f32();
      for (std::int64_t i = 0; i < I; ++i) {
        float gsum = 0.0f;
        float usum = 0.0f;
        for (std::int64_t d = 0; d < D; ++d) {
          const float xv = xb[static_cast<std::size_t>(d)];
          gsum += xv * Ge[static_cast<std::size_t>(i * D + d)];
          usum += xv * Ue[static_cast<std::size_t>(i * D + d)];
        }
        const float sig = 1.0f / (1.0f + std::exp(-gsum));
        hid[static_cast<std::size_t>(i)] = (gsum * sig) * usum;
      }
      for (std::int64_t o = 0; o < D; ++o) {
        float sum = 0.0f;
        for (std::int64_t i = 0; i < I; ++i) {
          sum += hid[static_cast<std::size_t>(i)] * De[static_cast<std::size_t>(o * I + i)];
        }
        Y[static_cast<std::size_t>(b * D + o)] += w * sum;
      }
    }
  }
  return Error::success();
}

Error reshape_f32(const TensorView& in, TensorView* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "reshape out");
  }
  if (in.numel() != out->numel()) {
    return Error::make(ErrorCode::InvalidArgument, "reshape numel mismatch");
  }
  if (in.data != out->data) {
    std::memcpy(out->data, in.data, in.nbytes);
  }
  return Error::success();
}

Error slice_f32(const TensorView& in,
                TensorView* out,
                std::int64_t start,
                std::int64_t end) {
  if (out == nullptr || in.data == nullptr || out->data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "slice null");
  }
  if (in.dtype != DType::F32 || out->dtype != DType::F32 || in.rank != 2 || out->rank != 2) {
    return Error::make(ErrorCode::InvalidArgument, "slice requires f32 rank-2");
  }
  if (in.dim(0) != out->dim(0)) {
    return Error::make(ErrorCode::InvalidArgument, "slice batch mismatch");
  }
  const std::int64_t in_dim = in.dim(1);
  const std::int64_t out_dim = out->dim(1);
  if (start < 0 || end <= start || end > in_dim || out_dim != end - start) {
    return Error::make(ErrorCode::InvalidArgument, "slice range/shape mismatch");
  }
  const float* src = in.f32();
  float* dst = out->f32();
  for (std::int64_t b = 0; b < in.dim(0); ++b) {
    std::memcpy(dst + static_cast<std::size_t>(b * out_dim),
                src + static_cast<std::size_t>(b * in_dim + start),
                static_cast<std::size_t>(out_dim) * sizeof(float));
  }
  return Error::success();
}

Error transpose_f32(const TensorView& in,
                    TensorView* out,
                    const std::vector<std::int64_t>& perm) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "transpose out");
  }
  if (in.rank != 2 || out->rank != 2 || perm.size() != 2) {
    return Error::make(ErrorCode::NotImplemented,
                       "transpose Phase 4 supports rank-2 only");
  }
  if (perm[0] == 0 && perm[1] == 1) {
    return identity_f32(in, out);
  }
  if (perm[0] != 1 || perm[1] != 0) {
    return Error::make(ErrorCode::InvalidArgument, "unsupported perm");
  }
  const std::int64_t rows = in.dim(0);
  const std::int64_t cols = in.dim(1);
  if (out->dim(0) != cols || out->dim(1) != rows) {
    return Error::make(ErrorCode::InvalidArgument, "transpose out shape");
  }
  const float* x = in.f32();
  float* y = out->f32();
  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < cols; ++j) {
      y[j * rows + i] = x[i * cols + j];
    }
  }
  return Error::success();
}

}  // namespace kernels
}  // namespace uaii
