#include "uaii/quant/gguf_dequant.hpp"

#include <cstring>

namespace uaii {
namespace quant {
namespace {

constexpr std::int64_t QK_K = 256;
constexpr std::size_t K_SCALE_SIZE = 12;

struct BlockQ40 {
  std::uint16_t d;
  std::uint8_t qs[16];
};
struct BlockQ41 {
  std::uint16_t d;
  std::uint16_t m;
  std::uint8_t qs[16];
};
struct BlockQ50 {
  std::uint16_t d;
  std::uint8_t qh[4];
  std::uint8_t qs[16];
};
struct BlockQ51 {
  std::uint16_t d;
  std::uint16_t m;
  std::uint8_t qh[4];
  std::uint8_t qs[16];
};
struct BlockQ80 {
  std::uint16_t d;
  std::int8_t qs[32];
};
struct BlockQ2K {
  std::uint8_t scales[16];
  std::uint8_t qs[64];
  std::uint16_t d;
  std::uint16_t dmin;
};
struct BlockQ3K {
  std::uint8_t hmask[32];
  std::uint8_t qs[64];
  std::uint8_t scales[12];
  std::uint16_t d;
};
struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[K_SCALE_SIZE];
  std::uint8_t qs[128];
};
struct BlockQ5K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[K_SCALE_SIZE];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
struct BlockQ6K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};

static_assert(sizeof(BlockQ40) == 18);
static_assert(sizeof(BlockQ41) == 20);
static_assert(sizeof(BlockQ50) == 22);
static_assert(sizeof(BlockQ51) == 24);
static_assert(sizeof(BlockQ80) == 34);
static_assert(sizeof(BlockQ2K) == 84);
static_assert(sizeof(BlockQ3K) == 110);
static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ5K) == 176);
static_assert(sizeof(BlockQ6K) == 210);

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u) << 16);
  std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t mant = h & 0x3ffu;
  std::uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      std::uint32_t m = mant;
      std::uint32_t e = 127 - 15 + 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3ffu;
      out = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

Error dequant_row_q4_0(const std::uint8_t* packed, std::int64_t cols, float* out) {
  constexpr std::int64_t QK = 32;
  if (cols % QK != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q4_0 cols not multiple of 32");
  }
  const auto* blocks = reinterpret_cast<const BlockQ40*>(packed);
  const std::int64_t nb = cols / QK;
  for (std::int64_t i = 0; i < nb; ++i) {
    const float d = f16_to_f32(blocks[i].d);
    for (std::int64_t j = 0; j < QK / 2; ++j) {
      const int x0 = (blocks[i].qs[j] & 0x0F) - 8;
      const int x1 = (blocks[i].qs[j] >> 4) - 8;
      out[i * QK + j] = static_cast<float>(x0) * d;
      out[i * QK + j + QK / 2] = static_cast<float>(x1) * d;
    }
  }
  return Error::success();
}

Error dequant_row_q4_1(const std::uint8_t* packed, std::int64_t cols, float* out) {
  constexpr std::int64_t QK = 32;
  if (cols % QK != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q4_1 cols not multiple of 32");
  }
  const auto* blocks = reinterpret_cast<const BlockQ41*>(packed);
  const std::int64_t nb = cols / QK;
  for (std::int64_t i = 0; i < nb; ++i) {
    const float d = f16_to_f32(blocks[i].d);
    const float m = f16_to_f32(blocks[i].m);
    for (std::int64_t j = 0; j < QK / 2; ++j) {
      const int x0 = blocks[i].qs[j] & 0x0F;
      const int x1 = blocks[i].qs[j] >> 4;
      out[i * QK + j] = static_cast<float>(x0) * d + m;
      out[i * QK + j + QK / 2] = static_cast<float>(x1) * d + m;
    }
  }
  return Error::success();
}

Error dequant_row_q5_0(const std::uint8_t* packed, std::int64_t cols, float* out) {
  constexpr std::int64_t QK = 32;
  if (cols % QK != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q5_0 cols not multiple of 32");
  }
  const auto* blocks = reinterpret_cast<const BlockQ50*>(packed);
  const std::int64_t nb = cols / QK;
  for (std::int64_t i = 0; i < nb; ++i) {
    const float d = f16_to_f32(blocks[i].d);
    std::uint32_t qh = 0;
    std::memcpy(&qh, blocks[i].qh, sizeof(qh));
    for (std::int64_t j = 0; j < QK / 2; ++j) {
      const std::uint8_t xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const std::uint8_t xh_1 = static_cast<std::uint8_t>(((qh >> (j + 12))) & 0x10);
      const int x0 = static_cast<int>((blocks[i].qs[j] & 0x0F) | xh_0) - 16;
      const int x1 = static_cast<int>((blocks[i].qs[j] >> 4) | xh_1) - 16;
      out[i * QK + j] = static_cast<float>(x0) * d;
      out[i * QK + j + QK / 2] = static_cast<float>(x1) * d;
    }
  }
  return Error::success();
}

Error dequant_row_q5_1(const std::uint8_t* packed, std::int64_t cols, float* out) {
  constexpr std::int64_t QK = 32;
  if (cols % QK != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q5_1 cols not multiple of 32");
  }
  const auto* blocks = reinterpret_cast<const BlockQ51*>(packed);
  const std::int64_t nb = cols / QK;
  for (std::int64_t i = 0; i < nb; ++i) {
    const float d = f16_to_f32(blocks[i].d);
    const float m = f16_to_f32(blocks[i].m);
    std::uint32_t qh = 0;
    std::memcpy(&qh, blocks[i].qh, sizeof(qh));
    for (std::int64_t j = 0; j < QK / 2; ++j) {
      const std::uint8_t xh_0 = static_cast<std::uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const std::uint8_t xh_1 = static_cast<std::uint8_t>(((qh >> (j + 12))) & 0x10);
      const int x0 = static_cast<int>((blocks[i].qs[j] & 0x0F) | xh_0);
      const int x1 = static_cast<int>((blocks[i].qs[j] >> 4) | xh_1);
      out[i * QK + j] = static_cast<float>(x0) * d + m;
      out[i * QK + j + QK / 2] = static_cast<float>(x1) * d + m;
    }
  }
  return Error::success();
}

Error dequant_row_q8_0(const std::uint8_t* packed, std::int64_t cols, float* out) {
  constexpr std::int64_t QK = 32;
  if (cols % QK != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q8_0 cols not multiple of 32");
  }
  const auto* blocks = reinterpret_cast<const BlockQ80*>(packed);
  const std::int64_t nb = cols / QK;
  for (std::int64_t i = 0; i < nb; ++i) {
    const float d = f16_to_f32(blocks[i].d);
    for (std::int64_t j = 0; j < QK; ++j) {
      out[i * QK + j] = static_cast<float>(blocks[i].qs[j]) * d;
    }
  }
  return Error::success();
}

Error dequant_row_q2_k(const std::uint8_t* packed, std::int64_t cols, float* out) {
  if (cols % QK_K != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q2_K cols not multiple of 256");
  }
  const auto* blocks = reinterpret_cast<const BlockQ2K*>(packed);
  const std::int64_t nb = cols / QK_K;
  for (std::int64_t i = 0; i < nb; ++i) {
    const float d = f16_to_f32(blocks[i].d);
    const float min = f16_to_f32(blocks[i].dmin);
    const std::uint8_t* q = blocks[i].qs;
    int is = 0;
    float* y = out + i * QK_K;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        std::uint8_t sc = blocks[i].scales[is++];
        float dl = d * static_cast<float>(sc & 0xF);
        float ml = min * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>(static_cast<std::int8_t>((q[l] >> shift) & 3)) - ml;
        }

        sc = blocks[i].scales[is++];
        dl = d * static_cast<float>(sc & 0xF);
        ml = min * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>(static_cast<std::int8_t>((q[l + 16] >> shift) & 3)) - ml;
        }

        shift += 2;
      }
      q += 32;
    }
  }
  return Error::success();
}

Error dequant_row_q3_k(const std::uint8_t* packed, std::int64_t cols, float* out) {
  if (cols % QK_K != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q3_K cols not multiple of 256");
  }
  const auto* blocks = reinterpret_cast<const BlockQ3K*>(packed);
  const std::int64_t nb = cols / QK_K;
  const std::uint32_t kmask1 = 0x03030303u;
  const std::uint32_t kmask2 = 0x0f0f0f0fu;

  for (std::int64_t i = 0; i < nb; ++i) {
    const float d_all = f16_to_f32(blocks[i].d);
    const std::uint8_t* q = blocks[i].qs;
    const std::uint8_t* hm = blocks[i].hmask;
    std::uint8_t m = 1;

    std::uint32_t aux[4];
    std::memcpy(aux, blocks[i].scales, 12);
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const auto* scales = reinterpret_cast<const std::int8_t*>(aux);

    int is = 0;
    float* y = out + i * QK_K;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>(
                     static_cast<std::int8_t>((q[l + 0] >> shift) & 3) -
                     ((hm[l + 0] & m) ? 0 : 4));
        }

        dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>(
                     static_cast<std::int8_t>((q[l + 16] >> shift) & 3) -
                     ((hm[l + 16] & m) ? 0 : 4));
        }

        shift += 2;
        m = static_cast<std::uint8_t>(m << 1);
      }
      q += 32;
    }
  }
  return Error::success();
}

Error dequant_row_q4_k(const std::uint8_t* packed, std::int64_t cols, float* out) {
  if (cols % QK_K != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q4_K cols not multiple of 256");
  }
  const auto* blocks = reinterpret_cast<const BlockQ4K*>(packed);
  const std::int64_t nb = cols / QK_K;
  for (std::int64_t i = 0; i < nb; ++i) {
    const std::uint8_t* q = blocks[i].qs;
    const float d = f16_to_f32(blocks[i].d);
    const float min = f16_to_f32(blocks[i].dmin);
    int is = 0;
    float* y = out + i * QK_K;
    for (int j = 0; j < QK_K; j += 64) {
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min * static_cast<float>(m);
      get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = min * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) {
        *y++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
      }
      for (int l = 0; l < 32; ++l) {
        *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
      }
      q += 32;
      is += 2;
    }
  }
  return Error::success();
}

Error dequant_row_q5_k(const std::uint8_t* packed, std::int64_t cols, float* out) {
  if (cols % QK_K != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q5_K cols not multiple of 256");
  }
  const auto* blocks = reinterpret_cast<const BlockQ5K*>(packed);
  const std::int64_t nb = cols / QK_K;
  for (std::int64_t i = 0; i < nb; ++i) {
    const std::uint8_t* ql = blocks[i].qs;
    const std::uint8_t* qh = blocks[i].qh;
    const float d = f16_to_f32(blocks[i].d);
    const float min = f16_to_f32(blocks[i].dmin);
    int is = 0;
    std::uint8_t u1 = 1;
    std::uint8_t u2 = 2;
    float* y = out + i * QK_K;
    for (int j = 0; j < QK_K; j += 64) {
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min * static_cast<float>(m);
      get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = min * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) {
        *y++ = d1 * static_cast<float>((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
      }
      for (int l = 0; l < 32; ++l) {
        *y++ = d2 * static_cast<float>((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
      }
      ql += 32;
      is += 2;
      u1 = static_cast<std::uint8_t>(u1 << 2);
      u2 = static_cast<std::uint8_t>(u2 << 2);
    }
  }
  return Error::success();
}

Error dequant_row_q6_k(const std::uint8_t* packed, std::int64_t cols, float* out) {
  if (cols % QK_K != 0) {
    return Error::make(ErrorCode::InvalidArgument, "Q6_K cols not multiple of 256");
  }
  const auto* blocks = reinterpret_cast<const BlockQ6K*>(packed);
  const std::int64_t nb = cols / QK_K;
  for (std::int64_t i = 0; i < nb; ++i) {
    const float d = f16_to_f32(blocks[i].d);
    const std::uint8_t* ql = blocks[i].ql;
    const std::uint8_t* qh = blocks[i].qh;
    const std::int8_t* sc = blocks[i].scales;
    float* y = out + i * QK_K;
    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const std::int8_t q1 =
            static_cast<std::int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const std::int8_t q2 =
            static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const std::int8_t q3 =
            static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const std::int8_t q4 =
            static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
  return Error::success();
}

}  // namespace

Error dequant_gguf_row(QuantFormat format, const std::uint8_t* packed, std::int64_t cols,
                       float* out) {
  if (packed == nullptr || out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "dequant_gguf_row null");
  }
  switch (format) {
    case QuantFormat::Q4_0: return dequant_row_q4_0(packed, cols, out);
    case QuantFormat::Q4_1: return dequant_row_q4_1(packed, cols, out);
    case QuantFormat::Q5_0: return dequant_row_q5_0(packed, cols, out);
    case QuantFormat::Q5_1: return dequant_row_q5_1(packed, cols, out);
    case QuantFormat::Q8_0: return dequant_row_q8_0(packed, cols, out);
    case QuantFormat::Q2_K: return dequant_row_q2_k(packed, cols, out);
    case QuantFormat::Q3_K: return dequant_row_q3_k(packed, cols, out);
    case QuantFormat::Q4_K: return dequant_row_q4_k(packed, cols, out);
    case QuantFormat::Q5_K: return dequant_row_q5_k(packed, cols, out);
    case QuantFormat::Q6_K: return dequant_row_q6_k(packed, cols, out);
    default:
      return Error::make(ErrorCode::NotImplemented, "unsupported gguf dequant format");
  }
}

}  // namespace quant
}  // namespace uaii
