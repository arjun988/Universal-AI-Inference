#pragma once

#include "uaii/core/error.hpp"
#include "uaii/export.hpp"
#include "uaii/quant/codec.hpp"

#include <memory>
#include <string>
#include <vector>

namespace uaii {
namespace quant {

/// Plugin-facing quantization interface (Phase 6).
class IQuantizer {
 public:
  virtual ~IQuantizer() = default;
  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual std::vector<QuantFormat> supported_formats() const = 0;
  [[nodiscard]] virtual Error pack(const float* src,
                                   std::size_t n,
                                   QuantFormat format,
                                   const QuantParams& params,
                                   std::vector<std::uint8_t>* packed,
                                   std::vector<float>* scales) = 0;
  [[nodiscard]] virtual Error unpack(const std::uint8_t* packed,
                                     std::size_t packed_nbytes,
                                     std::size_t n,
                                     QuantFormat format,
                                     const QuantParams& params,
                                     const float* scales,
                                     std::size_t n_scales,
                                     float* dst) = 0;
};

/// Built-in reference quantizer covering F16/BF16/INT8/INT4/NF4/MXFP4.
class UAII_API ReferenceQuantizer : public IQuantizer {
 public:
  [[nodiscard]] std::string name() const override { return "reference"; }
  [[nodiscard]] std::vector<QuantFormat> supported_formats() const override;
  [[nodiscard]] Error pack(const float* src,
                           std::size_t n,
                           QuantFormat format,
                           const QuantParams& params,
                           std::vector<std::uint8_t>* packed,
                           std::vector<float>* scales) override;
  [[nodiscard]] Error unpack(const std::uint8_t* packed,
                             std::size_t packed_nbytes,
                             std::size_t n,
                             QuantFormat format,
                             const QuantParams& params,
                             const float* scales,
                             std::size_t n_scales,
                             float* dst) override;
};

class UAII_API QuantizerRegistry {
 public:
  static QuantizerRegistry& instance();

  void register_quantizer(std::unique_ptr<IQuantizer> q);
  [[nodiscard]] IQuantizer* find(const std::string& name) noexcept;
  [[nodiscard]] IQuantizer* default_quantizer() noexcept;
  [[nodiscard]] std::vector<std::string> names() const;

 private:
  QuantizerRegistry();
  std::vector<std::unique_ptr<IQuantizer>> quantizers_;
};

}  // namespace quant
}  // namespace uaii
