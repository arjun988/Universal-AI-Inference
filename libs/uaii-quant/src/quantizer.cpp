#include "uaii/quant/quantizer.hpp"

namespace uaii {
namespace quant {

std::vector<QuantFormat> ReferenceQuantizer::supported_formats() const {
  return {QuantFormat::F32, QuantFormat::F16, QuantFormat::BF16, QuantFormat::INT8,
          QuantFormat::INT4, QuantFormat::NF4, QuantFormat::MXFP4};
}

Error ReferenceQuantizer::pack(const float* src,
                               std::size_t n,
                               QuantFormat format,
                               const QuantParams& params,
                               std::vector<std::uint8_t>* packed,
                               std::vector<float>* scales) {
  return pack_f32(src, n, format, params, packed, scales);
}

Error ReferenceQuantizer::unpack(const std::uint8_t* packed,
                                 std::size_t packed_nbytes,
                                 std::size_t n,
                                 QuantFormat format,
                                 const QuantParams& params,
                                 const float* scales,
                                 std::size_t n_scales,
                                 float* dst) {
  return unpack_to_f32(packed, packed_nbytes, n, format, params, scales, n_scales, dst);
}

QuantizerRegistry::QuantizerRegistry() {
  quantizers_.push_back(std::make_unique<ReferenceQuantizer>());
}

QuantizerRegistry& QuantizerRegistry::instance() {
  static QuantizerRegistry reg;
  return reg;
}

void QuantizerRegistry::register_quantizer(std::unique_ptr<IQuantizer> q) {
  if (q) quantizers_.push_back(std::move(q));
}

IQuantizer* QuantizerRegistry::find(const std::string& name) noexcept {
  for (auto& q : quantizers_) {
    if (q && q->name() == name) return q.get();
  }
  return nullptr;
}

IQuantizer* QuantizerRegistry::default_quantizer() noexcept {
  return quantizers_.empty() ? nullptr : quantizers_.front().get();
}

std::vector<std::string> QuantizerRegistry::names() const {
  std::vector<std::string> out;
  for (const auto& q : quantizers_) {
    if (q) out.push_back(q->name());
  }
  return out;
}

}  // namespace quant
}  // namespace uaii
