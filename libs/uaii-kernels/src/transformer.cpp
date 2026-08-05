#include "uaii/kernels/kernels.hpp"
#include "uaii/kernels/view_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace uaii {
namespace kernels {

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

Error attention_f32(const TensorView& q,
                    const TensorView& k,
                    const TensorView& v,
                    TensorView* out,
                    int num_heads,
                    float scale,
                    bool causal) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "attention out null");
  }
  // Rank-3: [batch, seq, dim]. Rank-2: [batch, dim] treated as seq=1.
  const bool rank2 = (q.rank == 2 && k.rank == 2 && v.rank == 2 && out->rank == 2);
  const bool rank3 = (q.rank == 3 && k.rank == 3 && v.rank == 3 && out->rank == 3);
  if (!rank2 && !rank3) {
    return Error::make(ErrorCode::InvalidArgument,
                       "Attention expects [batch,seq,dim] or [batch,dim]");
  }
  {
    Error err = check_view_bytes(q, "attn_q");
    if (!err.ok()) return err;
    err = check_view_bytes(k, "attn_k");
    if (!err.ok()) return err;
    err = check_view_bytes(v, "attn_v");
    if (!err.ok()) return err;
    err = check_view_bytes(*out, "attn_out");
    if (!err.ok()) return err;
  }
  const std::int64_t batch = q.dim(0);
  const std::int64_t seq = rank2 ? 1 : q.dim(1);
  const std::int64_t dim = rank2 ? q.dim(1) : q.dim(2);
  if (num_heads <= 0 || dim % num_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "invalid num_heads");
  }
  if (rank2) {
    if (k.dim(0) != batch || k.dim(1) != dim || v.dim(0) != batch || v.dim(1) != dim ||
        out->dim(0) != batch || out->dim(1) != dim) {
      return Error::make(ErrorCode::InvalidArgument, "attention shape mismatch");
    }
  } else if (k.dim(0) != batch || k.dim(1) != seq || k.dim(2) != dim ||
             v.dim(0) != batch || v.dim(1) != seq || v.dim(2) != dim ||
             out->dim(0) != batch || out->dim(1) != seq || out->dim(2) != dim) {
    return Error::make(ErrorCode::InvalidArgument, "attention shape mismatch");
  }
  if (scale <= 0) {
    scale = 1.0f / std::sqrt(static_cast<float>(dim / num_heads));
  }

  const std::int64_t head_dim = dim / num_heads;
  const float* Q = q.f32();
  const float* K = k.f32();
  const float* V = v.f32();
  float* O = out->f32();
  std::vector<float> scores(static_cast<std::size_t>(seq * seq));

  for (std::int64_t b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      // scores = softmax(QK^T * scale)
      for (std::int64_t i = 0; i < seq; ++i) {
        float max_v = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = 0; j < seq; ++j) {
          if (causal && j > i) {
            scores[static_cast<std::size_t>(i * seq + j)] = -1e9f;
            continue;
          }
          float dot = 0;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            const std::size_t qi =
                static_cast<std::size_t>(((b * seq + i) * dim) + h * head_dim + d);
            const std::size_t kj =
                static_cast<std::size_t>(((b * seq + j) * dim) + h * head_dim + d);
            dot += Q[qi] * K[kj];
          }
          dot *= scale;
          scores[static_cast<std::size_t>(i * seq + j)] = dot;
          max_v = std::max(max_v, dot);
        }
        float sum = 0;
        for (std::int64_t j = 0; j < seq; ++j) {
          float e = std::exp(scores[static_cast<std::size_t>(i * seq + j)] - max_v);
          scores[static_cast<std::size_t>(i * seq + j)] = e;
          sum += e;
        }
        const float inv = sum > 0 ? 1.0f / sum : 0.0f;
        for (std::int64_t j = 0; j < seq; ++j) {
          scores[static_cast<std::size_t>(i * seq + j)] *= inv;
        }
      }
      for (std::int64_t i = 0; i < seq; ++i) {
        for (std::int64_t d = 0; d < head_dim; ++d) {
          float acc = 0;
          for (std::int64_t j = 0; j < seq; ++j) {
            const std::size_t vj =
                static_cast<std::size_t>(((b * seq + j) * dim) + h * head_dim + d);
            acc += scores[static_cast<std::size_t>(i * seq + j)] * V[vj];
          }
          const std::size_t oi =
              static_cast<std::size_t>(((b * seq + i) * dim) + h * head_dim + d);
          O[oi] = acc;
        }
      }
    }
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
