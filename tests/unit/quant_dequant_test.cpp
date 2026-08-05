#include "uaii/quant/codec.hpp"
#include "uaii/quant/gguf_dequant.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

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

void pack_synthetic_q4_0(std::vector<std::uint8_t>* packed) {
  constexpr std::size_t QK = 32;
  packed->assign(18, 0);
  const std::uint16_t d_bits = f32_to_f16_bits(0.5f);
  std::memcpy(packed->data(), &d_bits, 2);
  for (std::size_t j = 0; j < QK / 2; ++j) {
    const int x0 = static_cast<int>((j + 1) % 8);
    const int x1 = static_cast<int>((j + 4) % 8);
    (*packed)[2 + j] = static_cast<std::uint8_t>((x0 & 0xF) | ((x1 & 0xF) << 4));
  }
}

bool near(float a, float b, float atol) {
  return std::fabs(a - b) <= atol;
}

void test_q4_0_dequant_parity() {
  std::vector<std::uint8_t> packed;
  pack_synthetic_q4_0(&packed);

  constexpr std::size_t cols = 32;
  std::vector<float> via_gemm(cols);
  std::vector<float> via_unpack(cols);

  const uaii::Error err_gemm =
      uaii::quant::dequant_gguf_row(uaii::quant::QuantFormat::Q4_0, packed.data(),
                                    static_cast<std::int64_t>(cols), via_gemm.data());
  expect(err_gemm.ok(), "dequant_gguf_row Q4_0");

  const uaii::quant::QuantParams params;
  const uaii::Error err_unpack = uaii::quant::unpack_to_f32(
      packed.data(), packed.size(), cols, uaii::quant::QuantFormat::Q4_0, params, nullptr, 0,
      via_unpack.data());
  expect(err_unpack.ok(), "unpack_to_f32 Q4_0");

  constexpr float atol = 1e-5f;
  for (std::size_t i = 0; i < cols; ++i) {
    if (!near(via_gemm[i], via_unpack[i], atol)) {
      std::cerr << "FAIL: Q4_0 parity at " << i << " gemm=" << via_gemm[i]
                << " unpack=" << via_unpack[i] << '\n';
      ++failures;
    }
  }
}

}  // namespace

int main() {
  test_q4_0_dequant_parity();

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "uaii_quant_dequant_test: OK\n";
  return EXIT_SUCCESS;
}
