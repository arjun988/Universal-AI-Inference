#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace uaii {
namespace runtime {

/// Autoregressive token sampling (best-practice stack for OSS LLMs).
/// temperature <= 0 → greedy argmax. Otherwise: optional top-k → nucleus (top-p) → softmax → sample.
struct SampleParams {
  float temperature = 0.0f;
  float top_p = 1.0f;       // nucleus; 1.0 = disabled
  std::int32_t top_k = 0;   // 0 = disabled
  float repetition_penalty = 1.0f;  // 1.0 = off; >1 penalizes tokens already in `penalty_tokens`
  std::uint64_t seed = 0;
  bool has_seed = false;

  [[nodiscard]] bool greedy() const noexcept {
    return temperature <= 0.0f && top_k <= 0 && top_p >= 1.0f - 1e-6f &&
           repetition_penalty <= 1.0f + 1e-6f;
  }
};

/// Sample one token id from a logits (or log-prob) row. `row` is length `vocab`.
/// When `params.greedy()`, returns argmax (RNG unused).
[[nodiscard]] UAII_API Error sample_token_f32(const float* row,
                                              std::size_t vocab,
                                              const SampleParams& params,
                                              const std::vector<std::int64_t>& penalty_tokens,
                                              std::mt19937_64* rng,
                                              std::int64_t* out_token);

}  // namespace runtime
}  // namespace uaii
