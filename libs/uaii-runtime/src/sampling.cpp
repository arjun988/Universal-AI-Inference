#include "uaii/runtime/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace uaii {
namespace runtime {
namespace {

float apply_rep_penalty(float logit, float penalty, bool seen) {
  if (!seen || penalty <= 1.0f + 1e-6f) return logit;
  // Common HF-style: divide positive logits, multiply negative.
  return logit > 0.0f ? logit / penalty : logit * penalty;
}

}  // namespace

Error sample_token_f32(const float* row,
                       std::size_t vocab,
                       const SampleParams& params,
                       const std::vector<std::int64_t>& penalty_tokens,
                       std::mt19937_64* rng,
                       std::int64_t* out_token) {
  if (row == nullptr || out_token == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "sample_token null");
  }
  if (vocab == 0) {
    return Error::make(ErrorCode::InvalidArgument, "sample_token empty vocab");
  }

  std::unordered_set<std::int64_t> seen(penalty_tokens.begin(), penalty_tokens.end());

  // Greedy path
  if (params.greedy()) {
    std::size_t best = 0;
    float best_v = apply_rep_penalty(row[0], params.repetition_penalty, seen.count(0) > 0);
    for (std::size_t i = 1; i < vocab; ++i) {
      const float v =
          apply_rep_penalty(row[i], params.repetition_penalty, seen.count(static_cast<std::int64_t>(i)) > 0);
      if (v > best_v) {
        best_v = v;
        best = i;
      }
    }
    *out_token = static_cast<std::int64_t>(best);
    return Error::success();
  }

  if (rng == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "sample_token requires rng");
  }

  std::vector<std::pair<float, std::int64_t>> scored;
  scored.reserve(vocab);
  const float inv_t = 1.0f / std::max(params.temperature, 1e-5f);
  for (std::size_t i = 0; i < vocab; ++i) {
    float v =
        apply_rep_penalty(row[i], params.repetition_penalty, seen.count(static_cast<std::int64_t>(i)) > 0);
    scored.emplace_back(v * inv_t, static_cast<std::int64_t>(i));
  }

  // Top-k: keep k largest logits
  if (params.top_k > 0 && static_cast<std::size_t>(params.top_k) < scored.size()) {
    const std::size_t k = static_cast<std::size_t>(params.top_k);
    std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k), scored.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    scored.resize(k);
  }

  // Softmax numerically stable
  float max_v = -std::numeric_limits<float>::infinity();
  for (const auto& s : scored) max_v = std::max(max_v, s.first);
  float sum = 0.0f;
  for (auto& s : scored) {
    s.first = std::exp(s.first - max_v);
    sum += s.first;
  }
  if (!(sum > 0.0f) || !std::isfinite(sum)) {
    // Degenerate → argmax of remaining
    auto it = std::max_element(scored.begin(), scored.end(),
                               [](const auto& a, const auto& b) { return a.first < b.first; });
    *out_token = it->second;
    return Error::success();
  }
  for (auto& s : scored) s.first /= sum;

  // Nucleus (top-p): sort by prob desc, keep cumulative mass
  if (params.top_p > 0.0f && params.top_p < 1.0f - 1e-6f) {
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    float cum = 0.0f;
    std::size_t keep = 0;
    for (; keep < scored.size(); ++keep) {
      cum += scored[keep].first;
      if (cum >= params.top_p) {
        ++keep;
        break;
      }
    }
    if (keep == 0) keep = 1;
    scored.resize(keep);
    float renorm = 0.0f;
    for (const auto& s : scored) renorm += s.first;
    if (renorm > 0.0f) {
      for (auto& s : scored) s.first /= renorm;
    }
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(*rng);
  float cum = 0.0f;
  for (const auto& s : scored) {
    cum += s.first;
    if (r <= cum) {
      *out_token = s.second;
      return Error::success();
    }
  }
  *out_token = scored.back().second;
  return Error::success();
}

}  // namespace runtime
}  // namespace uaii
