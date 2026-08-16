#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/interfaces/loader.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace loaders {

/// Strict Hugging Face decoder-only config (config.json). Critical fields fail closed.
struct HfConfig {
  std::string architecture;
  std::string model_type;
  std::int64_t hidden_size = 0;
  std::int64_t num_hidden_layers = 0;
  std::int64_t num_attention_heads = 0;
  std::int64_t num_key_value_heads = 0;
  std::int64_t intermediate_size = 0;
  std::int64_t vocab_size = 0;
  std::int64_t max_position_embeddings = 0;
  std::int64_t head_dim = 0;
  float rms_norm_eps = 0.f;
  float rope_theta = 0.f;
  int rope_mode = 0;  // 0=interleaved, 2=NeoX
  bool tie_word_embeddings = false;
  std::int64_t num_experts = 0;
  std::int64_t num_experts_per_tok = 0;
  std::string hidden_act;  // silu / gelu / ...
  std::string chat_template;  // optional from tokenizer_config.json
  /// rope_scaling.rope_type / type: empty, linear, llama3, dynamic, yarn.
  std::string rope_scaling_type;
  float rope_scale_factor = 1.f;
  float rope_low_freq_factor = 1.f;
  float rope_high_freq_factor = 4.f;
  float rope_original_max_pos = 0.f;
  float rope_attn_factor = 1.f;
  std::string quant_method;
  bool mxfp4 = false;
};

struct HfCapability {
  std::string family;
  std::string architecture;
  std::string model_type;
  std::string weight_dtype;
  std::string tokenizer_kind;
  std::string rope_scaling_type;
  bool moe = false;
  bool mla = false;
  bool quantized = false;
  bool mxfp4 = false;
  std::vector<std::string> blocked;
  std::vector<std::string> notes;
  [[nodiscard]] bool ok() const { return blocked.empty(); }
};

/// Probe a Hub folder without generating. blocked[] is what generate would throw.
[[nodiscard]] UAII_API Error hf_inspect_directory(const std::string& dir, HfCapability* out);

/// Unique shard filenames from a Hub weight index JSON (`weight_map` values).
[[nodiscard]] UAII_API Error hf_index_unique_shards(const std::string& index_json,
                                                    std::vector<std::string>* shards);

/// Map architecture / model_type to the product family name (llama, qwen2, gpt_oss, …).
[[nodiscard]] UAII_API std::string hf_family_name(const std::string& architecture,
                                                  const std::string& model_type) noexcept;

/// Human-readable capability report (`uaii inspect` / `generate --dry-run`).
[[nodiscard]] UAII_API std::string hf_format_capability(const HfCapability& cap);

[[nodiscard]] UAII_API bool hf_architecture_supported(const std::string& arch_name) noexcept;

[[nodiscard]] UAII_API Error hf_parse_config_json(const std::string& text, HfConfig* out);

[[nodiscard]] UAII_API Error hf_load_config_file(const std::string& path, HfConfig* out);

/// weight_map: tensor name → relative shard filename
[[nodiscard]] UAII_API Error hf_load_weight_index(
    const std::string& dir,
    std::unordered_map<std::string, std::string>* weight_map,
    std::vector<std::string>* shard_files);

/// Directory loader: config.json + tokenizer* + *.safetensors (± shards).
class UAII_API HfLoader : public IModelLoader {
 public:
  [[nodiscard]] LoaderInfo info() const override;
  [[nodiscard]] bool accepts(const std::string& path) const override;
  [[nodiscard]] Error load(const std::string& path, ir::Graph* out_graph) override;
};

}  // namespace loaders
}  // namespace uaii
