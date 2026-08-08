#include "uaii/kernels/kernels.hpp"
#include "uaii/kernels/view_util.hpp"
#include "uaii/runtime/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

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
  if (tokens.dtype != DType::F32 || weight.dtype != DType::F32 ||
      out->dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "embedding requires f32");
  }
  if (weight.rank != 2 || tokens.rank != 2) {
    return Error::make(ErrorCode::InvalidArgument, "embedding rank (tokens/weight)");
  }
  Error err = check_view_bytes(tokens, "tokens");
  if (!err.ok()) return err;
  err = check_view_bytes(weight, "weight");
  if (!err.ok()) return err;

  const std::int64_t vocab = weight.dim(0);
  const std::int64_t dim = weight.dim(1);
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
  const float* w = weight.f32();
  float* y = out->f32();
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t s = 0; s < seq; ++s) {
      const int id = static_cast<int>(ids[b * seq + s]);
      if (id < 0 || id >= static_cast<int>(vocab)) {
        return Error::make(ErrorCode::InvalidArgument,
                           "embedding token id out of range: " + std::to_string(id) +
                               " vocab=" + std::to_string(vocab));
      }
      const std::int64_t row = b * seq + s;
      std::memcpy(y + row * dim, w + static_cast<std::int64_t>(id) * dim,
                  static_cast<std::size_t>(dim) * sizeof(float));
    }
  }
  return Error::success();
}

Error rope_f32(const TensorView& in,
               const TensorView* positions,
               TensorView* out,
               float theta) {
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
  std::size_t rows = in.numel() / static_cast<std::size_t>(dim);
  const float* x = in.f32();
  float* y = out->f32();
  for (std::size_t r = 0; r < rows; ++r) {
    float pos = static_cast<float>(r);
    if (positions && positions->data && positions->numel() > r) {
      pos = positions->f32()[r];
    }
    const float* row = x + r * static_cast<std::size_t>(dim);
    float* out_row = y + r * static_cast<std::size_t>(dim);
    for (std::int64_t i = 0; i < dim; i += 2) {
      const float freq =
          1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(dim));
      const float angle = pos * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float a = row[i];
      const float b = row[i + 1];
      out_row[i] = a * c - b * s;
      out_row[i + 1] = a * s + b * c;
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
                       TensorView* present_v) {
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

  // Q/out share q_dim; K/V share kv_dim (GQA: kv_dim may be < q_dim).
  if (qs.batch != ks.batch || qs.batch != vs.batch || qs.batch != os.batch ||
      qs.dim != os.dim || ks.dim != vs.dim || qs.seq != os.seq || ks.seq != vs.seq ||
      qs.seq != ks.seq) {
    return Error::make(ErrorCode::InvalidArgument, "attention shape mismatch");
  }

  const std::int64_t batch = qs.batch;
  const std::int64_t q_seq = qs.seq;
  const std::int64_t new_kv_seq = ks.seq;
  const std::int64_t q_dim = qs.dim;
  const std::int64_t kv_dim = ks.dim;
  if (num_heads <= 0 || q_dim % num_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "invalid num_heads");
  }
  const std::int64_t head_dim = q_dim / num_heads;
  if (kv_dim % head_dim != 0) {
    return Error::make(ErrorCode::InvalidArgument, "kv_dim not divisible by head_dim");
  }
  const int num_kv_heads = static_cast<int>(kv_dim / head_dim);
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "invalid GQA head grouping");
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
    if (pks.batch != batch || pvs.batch != batch || pks.dim != kv_dim || pvs.dim != kv_dim ||
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
      return PK[static_cast<std::size_t>(((b * past_seq + j) * kv_dim) + kv_h * head_dim + d)];
    }
    const std::int64_t jj = j - past_seq;
    return K[static_cast<std::size_t>(((b * new_kv_seq + jj) * kv_dim) + kv_h * head_dim + d)];
  };
  auto v_at = [&](std::int64_t b, std::int64_t j, int kv_h, std::int64_t d) -> float {
    if (j < past_seq) {
      return PV[static_cast<std::size_t>(((b * past_seq + j) * kv_dim) + kv_h * head_dim + d)];
    }
    const std::int64_t jj = j - past_seq;
    return V[static_cast<std::size_t>(((b * new_kv_seq + jj) * kv_dim) + kv_h * head_dim + d)];
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
          float dot = 0;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            const std::size_t qi =
                static_cast<std::size_t>(((b * q_seq + i) * q_dim) + h * head_dim + d);
            dot += Q[qi] * k_at(b, j, kv_h, d);
          }
          dot *= scale;
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
        for (std::int64_t d = 0; d < head_dim; ++d) {
          float acc = 0;
          for (std::int64_t j = 0; j < kv_seq; ++j) {
            acc += scores[static_cast<std::size_t>(i * kv_seq + j)] * v_at(b, j, kv_h, d);
          }
          const std::size_t oi =
              static_cast<std::size_t>(((b * q_seq + i) * q_dim) + h * head_dim + d);
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
    const std::size_t need =
        static_cast<std::size_t>(batch * kv_seq * kv_dim) * sizeof(float);
    if (present_k->nbytes < need || present_v->nbytes < need) {
      return Error::make(ErrorCode::InvalidArgument, "present buffer too small");
    }
    float* pk = present_k->f32();
    float* pv = present_v->f32();
    // Stride for in-place capacity buffers uses max capacity inferred from nbytes.
    const std::int64_t stride_seq =
        inplace ? static_cast<std::int64_t>(present_k->nbytes /
                                           (static_cast<std::size_t>(batch * kv_dim) *
                                            sizeof(float)))
                : kv_seq;
    if (stride_seq < kv_seq) {
      return Error::make(ErrorCode::InvalidArgument, "present stride < kv_seq");
    }
    if (!inplace) {
      for (std::int64_t b = 0; b < batch; ++b) {
        if (past_seq > 0) {
          std::memcpy(pk + static_cast<std::size_t>(b * kv_seq * kv_dim),
                      PK + static_cast<std::size_t>(b * past_seq * kv_dim),
                      static_cast<std::size_t>(past_seq * kv_dim) * sizeof(float));
          std::memcpy(pv + static_cast<std::size_t>(b * kv_seq * kv_dim),
                      PV + static_cast<std::size_t>(b * past_seq * kv_dim),
                      static_cast<std::size_t>(past_seq * kv_dim) * sizeof(float));
        }
        std::memcpy(pk + static_cast<std::size_t>((b * kv_seq + past_seq) * kv_dim),
                    K + static_cast<std::size_t>(b * new_kv_seq * kv_dim),
                    static_cast<std::size_t>(new_kv_seq * kv_dim) * sizeof(float));
        std::memcpy(pv + static_cast<std::size_t>((b * kv_seq + past_seq) * kv_dim),
                    V + static_cast<std::size_t>(b * new_kv_seq * kv_dim),
                    static_cast<std::size_t>(new_kv_seq * kv_dim) * sizeof(float));
      }
    } else {
      for (std::int64_t b = 0; b < batch; ++b) {
        for (std::int64_t s = 0; s < new_kv_seq; ++s) {
          const std::size_t src =
              static_cast<std::size_t>((b * new_kv_seq + s) * kv_dim);
          const std::size_t dst =
              static_cast<std::size_t>((b * stride_seq + past_seq + s) * kv_dim);
          std::memcpy(pk + dst, K + src, static_cast<std::size_t>(kv_dim) * sizeof(float));
          std::memcpy(pv + dst, V + src, static_cast<std::size_t>(kv_dim) * sizeof(float));
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
                    bool causal) {
  return attention_kv_f32(q, k, v, out, num_heads, scale, causal, nullptr, nullptr,
                          nullptr, nullptr);
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
