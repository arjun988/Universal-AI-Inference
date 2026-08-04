#include "uaii/quant/codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace uaii {
namespace quant {
namespace {

std::uint16_t f32_to_f16_bits(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xff) - 127 + 15;
  std::uint32_t mant = x & 0x7fffffu;
  if (exp <= 0) {
    return static_cast<std::uint16_t>(sign);
  }
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                   (mant >> 13));
}

float f16_bits_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u) << 16);
  std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t mant = h & 0x3ffu;
  std::uint32_t out;
  if (exp == 0) {
    out = sign;
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

std::uint16_t f32_to_bf16_bits(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  return static_cast<std::uint16_t>(x >> 16);
}

float bf16_bits_to_f32(std::uint16_t h) {
  std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

// NF4 codebook (bitsandbytes-style approx)
constexpr float kNf4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f,
};

int nearest_nf4(float x) {
  int best = 0;
  float best_d = std::fabs(x - kNf4[0]);
  for (int i = 1; i < 16; ++i) {
    const float d = std::fabs(x - kNf4[i]);
    if (d < best_d) {
      best_d = d;
      best = i;
    }
  }
  return best;
}

}  // namespace

Error pack_f32(const float* src,
               std::size_t n,
               QuantFormat format,
               const QuantParams& params,
               std::vector<std::uint8_t>* packed,
               std::vector<float>* scales) {
  if (src == nullptr || packed == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "pack args null");
  }
  packed->clear();
  if (scales) scales->clear();

  switch (format) {
    case QuantFormat::F32: {
      packed->resize(n * 4);
      std::memcpy(packed->data(), src, n * 4);
      return Error::ok();
    }
    case QuantFormat::F16: {
      packed->resize(n * 2);
      auto* dst = reinterpret_cast<std::uint16_t*>(packed->data());
      for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_f16_bits(src[i]);
      return Error::ok();
    }
    case QuantFormat::BF16: {
      packed->resize(n * 2);
      auto* dst = reinterpret_cast<std::uint16_t*>(packed->data());
      for (std::size_t i = 0; i < n; ++i) dst[i] = f32_to_bf16_bits(src[i]);
      return Error::ok();
    }
    case QuantFormat::INT8: {
      float scale = params.scale;
      if (scale == 0.f) {
        float max_abs = 0.f;
        for (std::size_t i = 0; i < n; ++i) max_abs = std::max(max_abs, std::fabs(src[i]));
        scale = (max_abs > 0.f) ? (max_abs / 127.f) : 1.f;
      }
      if (scales) scales->push_back(scale);
      packed->resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        float q = src[i] / scale + params.zero_point;
        q = std::max(-128.f, std::min(127.f, std::nearbyintf(q)));
        (*packed)[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
      }
      return Error::ok();
    }
    case QuantFormat::INT4: {
      float scale = params.scale;
      if (scale == 0.f) {
        float max_abs = 0.f;
        for (std::size_t i = 0; i < n; ++i) max_abs = std::max(max_abs, std::fabs(src[i]));
        scale = (max_abs > 0.f) ? (max_abs / 7.f) : 1.f;
      }
      if (scales) scales->push_back(scale);
      packed->assign((n + 1) / 2, 0);
      for (std::size_t i = 0; i < n; ++i) {
        float q = src[i] / scale;
        int v = static_cast<int>(std::max(-8.f, std::min(7.f, std::nearbyintf(q))));
        const std::uint8_t nibble = static_cast<std::uint8_t>(v & 0xf);
        if ((i & 1u) == 0) (*packed)[i / 2] = nibble;
        else (*packed)[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
      }
      return Error::ok();
    }
    case QuantFormat::NF4:
    case QuantFormat::MXFP4: {
      const int gs = params.group_size > 0 ? params.group_size : 32;
      if (scales) {
        for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(gs)) {
          float max_abs = 0.f;
          const std::size_t end = std::min(n, i + static_cast<std::size_t>(gs));
          for (std::size_t j = i; j < end; ++j) max_abs = std::max(max_abs, std::fabs(src[j]));
          scales->push_back(max_abs > 0.f ? max_abs : 1.f);
        }
      }
      packed->assign((n + 1) / 2, 0);
      for (std::size_t i = 0; i < n; ++i) {
        const float scale =
            (scales && !scales->empty())
                ? (*scales)[i / static_cast<std::size_t>(gs)]
                : (params.scale != 0.f ? params.scale : 1.f);
        const float x = src[i] / scale;
        int idx = 0;
        if (format == QuantFormat::NF4) {
          idx = nearest_nf4(std::max(-1.f, std::min(1.f, x)));
        } else {
          // MXFP4 scaffold: 4-bit float-ish via NF4 codebook
          idx = nearest_nf4(std::max(-1.f, std::min(1.f, x)));
        }
        const std::uint8_t nibble = static_cast<std::uint8_t>(idx & 0xf);
        if ((i & 1u) == 0) (*packed)[i / 2] = nibble;
        else (*packed)[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
      }
      return Error::ok();
    }
    default:
      return Error::make(ErrorCode::NotImplemented, "unsupported quant format");
  }
}

Error unpack_to_f32(const std::uint8_t* packed,
                    std::size_t packed_nbytes,
                    std::size_t n,
                    QuantFormat format,
                    const QuantParams& params,
                    const float* scales,
                    std::size_t n_scales,
                    float* dst) {
  if (packed == nullptr || dst == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "unpack args null");
  }
  (void)packed_nbytes;

  switch (format) {
    case QuantFormat::F32:
      std::memcpy(dst, packed, n * 4);
      return Error::ok();
    case QuantFormat::F16: {
      const auto* src = reinterpret_cast<const std::uint16_t*>(packed);
      for (std::size_t i = 0; i < n; ++i) dst[i] = f16_bits_to_f32(src[i]);
      return Error::ok();
    }
    case QuantFormat::BF16: {
      const auto* src = reinterpret_cast<const std::uint16_t*>(packed);
      for (std::size_t i = 0; i < n; ++i) dst[i] = bf16_bits_to_f32(src[i]);
      return Error::ok();
    }
    case QuantFormat::INT8: {
      const float scale = (scales && n_scales > 0) ? scales[0] : (params.scale != 0.f ? params.scale : 1.f);
      for (std::size_t i = 0; i < n; ++i) {
        const auto q = static_cast<std::int8_t>(packed[i]);
        dst[i] = (static_cast<float>(q) - params.zero_point) * scale;
      }
      return Error::ok();
    }
    case QuantFormat::INT4: {
      const float scale = (scales && n_scales > 0) ? scales[0] : (params.scale != 0.f ? params.scale : 1.f);
      for (std::size_t i = 0; i < n; ++i) {
        std::uint8_t nibble = packed[i / 2];
        nibble = ((i & 1u) == 0) ? (nibble & 0xfu) : (nibble >> 4);
        int v = static_cast<int>(nibble);
        if (v >= 8) v -= 16;
        dst[i] = static_cast<float>(v) * scale;
      }
      return Error::ok();
    }
    case QuantFormat::NF4:
    case QuantFormat::MXFP4: {
      const int gs = params.group_size > 0 ? params.group_size : 32;
      for (std::size_t i = 0; i < n; ++i) {
        std::uint8_t nibble = packed[i / 2];
        nibble = ((i & 1u) == 0) ? (nibble & 0xfu) : (nibble >> 4);
        const float scale =
            (scales && n_scales > 0)
                ? scales[std::min(n_scales - 1, i / static_cast<std::size_t>(gs))]
                : 1.f;
        dst[i] = kNf4[nibble & 0xf] * scale;
      }
      return Error::ok();
    }
    default:
      return Error::make(ErrorCode::NotImplemented, "unsupported quant format");
  }
}

}  // namespace quant
}  // namespace uaii
