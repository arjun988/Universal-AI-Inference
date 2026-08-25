#include "uaii/loaders/gguf.hpp"

#include "binary_io.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/quant/formats.hpp"
#include "uaii/quant/gguf_dequant.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

namespace uaii {
namespace loaders {
namespace {

enum class GgufValueType : std::uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

Error read_value(detail::ByteReader* r, GgufValueType type, GgufValue* out);

Error read_array(detail::ByteReader* r, GgufValue* out) {
  std::uint32_t et = 0;
  Error err = r->read_pod(&et);
  if (!err.ok()) return err;
  std::uint64_t n = 0;
  err = r->read_pod(&n);
  if (!err.ok()) return err;

  const auto at = static_cast<GgufValueType>(et);
  if (at == GgufValueType::String) {
    std::vector<std::string> vals;
    vals.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      std::string s;
      err = r->read_string(&s);
      if (!err.ok()) return err;
      vals.push_back(std::move(s));
    }
    *out = std::move(vals);
    return Error::success();
  }
  if (at == GgufValueType::Int32) {
    std::vector<std::int32_t> vals(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      err = r->read_pod(&vals[static_cast<std::size_t>(i)]);
      if (!err.ok()) return err;
    }
    *out = std::move(vals);
    return Error::success();
  }
  if (at == GgufValueType::Int64) {
    std::vector<std::int64_t> vals(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      err = r->read_pod(&vals[static_cast<std::size_t>(i)]);
      if (!err.ok()) return err;
    }
    *out = std::move(vals);
    return Error::success();
  }
  if (at == GgufValueType::Float32) {
    std::vector<float> vals(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
      err = r->read_pod(&vals[static_cast<std::size_t>(i)]);
      if (!err.ok()) return err;
    }
    *out = std::move(vals);
    return Error::success();
  }

  // Skip unsupported array element types by reading scalars into discard.
  for (std::uint64_t i = 0; i < n; ++i) {
    GgufValue tmp;
    err = read_value(r, at, &tmp);
    if (!err.ok()) return err;
  }
  *out = std::monostate{};
  return Error::success();
}

Error read_value(detail::ByteReader* r, GgufValueType type, GgufValue* out) {
  switch (type) {
    case GgufValueType::Uint8: {
      std::uint8_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Int8: {
      std::int8_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Uint16: {
      std::uint16_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Int16: {
      std::int16_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Uint32: {
      std::uint32_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Int32: {
      std::int32_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Float32: {
      float v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Bool: {
      std::int8_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = (v != 0);
      return Error::success();
    }
    case GgufValueType::String: {
      std::string s;
      Error err = r->read_string(&s);
      if (!err.ok()) return err;
      *out = std::move(s);
      return Error::success();
    }
    case GgufValueType::Array:
      return read_array(r, out);
    case GgufValueType::Uint64: {
      std::uint64_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Int64: {
      std::int64_t v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
    case GgufValueType::Float64: {
      double v = 0;
      Error err = r->read_pod(&v);
      if (!err.ok()) return err;
      *out = v;
      return Error::success();
    }
  }
  return Error::make(ErrorCode::InvalidArgument, "unknown GGUF value type");
}

void write_value(detail::ByteWriter* w, GgufValueType type, const GgufValue& value) {
  w->write_pod(static_cast<std::uint32_t>(type));
  switch (type) {
    case GgufValueType::Uint32:
      w->write_pod(std::get<std::uint32_t>(value));
      break;
    case GgufValueType::Int32:
      w->write_pod(std::get<std::int32_t>(value));
      break;
    case GgufValueType::Float32:
      w->write_pod(std::get<float>(value));
      break;
    case GgufValueType::Bool:
      w->write_pod(static_cast<std::int8_t>(std::get<bool>(value) ? 1 : 0));
      break;
    case GgufValueType::String:
      w->write_string(std::get<std::string>(value));
      break;
    case GgufValueType::Uint64:
      w->write_pod(std::get<std::uint64_t>(value));
      break;
    case GgufValueType::Int64:
      w->write_pod(std::get<std::int64_t>(value));
      break;
    default:
      break;
  }
}

void write_array(detail::ByteWriter* w, GgufValueType elem_type, const GgufValue& value) {
  w->write_pod(static_cast<std::uint32_t>(GgufValueType::Array));
  w->write_pod(static_cast<std::uint32_t>(elem_type));
  if (std::holds_alternative<std::vector<std::string>>(value)) {
    const auto& vals = std::get<std::vector<std::string>>(value);
    w->write_pod(static_cast<std::uint64_t>(vals.size()));
    for (const auto& s : vals) w->write_string(s);
    return;
  }
  if (std::holds_alternative<std::vector<std::int32_t>>(value)) {
    const auto& vals = std::get<std::vector<std::int32_t>>(value);
    w->write_pod(static_cast<std::uint64_t>(vals.size()));
    for (auto v : vals) w->write_pod(v);
    return;
  }
  if (std::holds_alternative<std::vector<std::int64_t>>(value)) {
    const auto& vals = std::get<std::vector<std::int64_t>>(value);
    w->write_pod(static_cast<std::uint64_t>(vals.size()));
    for (auto v : vals) w->write_pod(v);
    return;
  }
  if (std::holds_alternative<std::vector<float>>(value)) {
    const auto& vals = std::get<std::vector<float>>(value);
    w->write_pod(static_cast<std::uint64_t>(vals.size()));
    for (auto v : vals) w->write_pod(v);
    return;
  }
  w->write_pod(static_cast<std::uint64_t>(0));
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1u;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;
  std::uint32_t fbits = 0;
  if (exp == 0) {
    if (mant == 0) {
      fbits = sign << 31;
    } else {
      // subnormal
      float f = std::ldexp(static_cast<float>(mant), -24);
      if (sign) f = -f;
      return f;
    }
  } else if (exp == 31) {
    fbits = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    fbits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  }
  float out = 0;
  std::memcpy(&out, &fbits, sizeof(out));
  return out;
}

Shape dims_to_shape(const std::vector<std::uint64_t>& dims) {
  // GGUF stores dimensions in reverse order relative to typical row-major [rows, cols].
  Shape s;
  s.dims.reserve(dims.size());
  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    s.dims.push_back(static_cast<std::int64_t>(*it));
  }
  return s;
}

std::size_t numel_of(const std::vector<std::uint64_t>& dims) {
  std::size_t n = 1;
  for (auto d : dims) n *= static_cast<std::size_t>(d);
  return n;
}

/// `name#e=3` → expert 3 of a packed Mixtral/Qwen GGUF bank (outermost GGUF dim).
int parse_expert_slice(std::string* name) {
  if (name == nullptr) return -1;
  const auto pos = name->find("#e=");
  if (pos == std::string::npos) return -1;
  const int e = std::atoi(name->c_str() + static_cast<std::ptrdiff_t>(pos + 3));
  name->resize(pos);
  return e;
}

}  // namespace

const char* to_string(GgufType type) noexcept {
  switch (type) {
    case GgufType::F32: return "f32";
    case GgufType::F16: return "f16";
    case GgufType::Q4_0: return "q4_0";
    case GgufType::Q4_1: return "q4_1";
    case GgufType::Q5_0: return "q5_0";
    case GgufType::Q5_1: return "q5_1";
    case GgufType::Q8_0: return "q8_0";
    case GgufType::Q2_K: return "q2_k";
    case GgufType::Q3_K: return "q3_k";
    case GgufType::Q4_K: return "q4_k";
    case GgufType::Q5_K: return "q5_k";
    case GgufType::Q6_K: return "q6_k";
    case GgufType::Q8_K: return "q8_k";
    case GgufType::IQ2_XXS: return "iq2_xxs";
    case GgufType::IQ2_XS: return "iq2_xs";
    case GgufType::IQ3_XXS: return "iq3_xxs";
    case GgufType::IQ1_S: return "iq1_s";
    case GgufType::IQ4_NL: return "iq4_nl";
    case GgufType::IQ3_S: return "iq3_s";
    case GgufType::IQ2_S: return "iq2_s";
    case GgufType::IQ4_XS: return "iq4_xs";
    case GgufType::I8: return "i8";
    case GgufType::I16: return "i16";
    case GgufType::I32: return "i32";
    case GgufType::I64: return "i64";
    case GgufType::F64: return "f64";
    case GgufType::IQ1_M: return "iq1_m";
    case GgufType::BF16: return "bf16";
    default: return "other";
  }
}

bool gguf_type_supported(GgufType t) noexcept {
  switch (t) {
    case GgufType::F32:
    case GgufType::F16:
    case GgufType::Q4_0:
    case GgufType::Q4_1:
    case GgufType::Q5_0:
    case GgufType::Q5_1:
    case GgufType::Q8_0:
    case GgufType::Q2_K:
    case GgufType::Q3_K:
    case GgufType::Q4_K:
    case GgufType::Q5_K:
    case GgufType::Q6_K:
    case GgufType::IQ4_NL:
      return true;
    default:
      return false;
  }
}

quant::QuantFormat gguf_type_to_quant(GgufType t) noexcept {
  switch (t) {
    case GgufType::F32: return quant::QuantFormat::F32;
    case GgufType::F16: return quant::QuantFormat::F16;
    case GgufType::Q4_0: return quant::QuantFormat::Q4_0;
    case GgufType::Q4_1: return quant::QuantFormat::Q4_1;
    case GgufType::Q5_0: return quant::QuantFormat::Q5_0;
    case GgufType::Q5_1: return quant::QuantFormat::Q5_1;
    case GgufType::Q8_0: return quant::QuantFormat::Q8_0;
    case GgufType::Q2_K: return quant::QuantFormat::Q2_K;
    case GgufType::Q3_K: return quant::QuantFormat::Q3_K;
    case GgufType::Q4_K: return quant::QuantFormat::Q4_K;
    case GgufType::Q5_K: return quant::QuantFormat::Q5_K;
    case GgufType::Q6_K: return quant::QuantFormat::Q6_K;
    case GgufType::IQ4_NL: return quant::QuantFormat::IQ4_NL;
    default: break;
  }
  return quant::QuantFormat::F32;
}

DType gguf_to_dtype(GgufType type) noexcept {
  switch (type) {
    case GgufType::F32: return DType::F32;
    case GgufType::F16: return DType::F16;
    case GgufType::BF16: return DType::BF16;
    case GgufType::I8: return DType::I8;
    case GgufType::I32: return DType::I32;
    case GgufType::I64: return DType::I64;
    default: return DType::Unknown;
  }
}

std::size_t gguf_type_nbytes_per_elem(GgufType type) noexcept {
  switch (type) {
    case GgufType::F32: return 4;
    case GgufType::F16: return 2;
    case GgufType::BF16: return 2;
    case GgufType::I8: return 1;
    case GgufType::I16: return 2;
    case GgufType::I32: return 4;
    case GgufType::I64: return 8;
    case GgufType::F64: return 8;
    default: return 0;
  }
}

Error gguf_read_header(const std::string& path, GgufFile* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "gguf out null");
  }
  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(path, &r);
  if (!err.ok()) return err;

  char magic[4]{};
  err = r.read_bytes(magic, 4);
  if (!err.ok()) return err;
  if (std::strncmp(magic, "GGUF", 4) != 0) {
    return Error::make(ErrorCode::InvalidArgument, "not a GGUF file");
  }

  GgufFile file;
  file.path = path;
  err = r.read_pod(&file.version);
  if (!err.ok()) return err;

  std::uint64_t tensor_count = 0;
  std::uint64_t kv_count = 0;
  err = r.read_pod(&tensor_count);
  if (!err.ok()) return err;
  err = r.read_pod(&kv_count);
  if (!err.ok()) return err;

  for (std::uint64_t i = 0; i < kv_count; ++i) {
    std::string key;
    err = r.read_string(&key);
    if (!err.ok()) return err;
    std::uint32_t vt = 0;
    err = r.read_pod(&vt);
    if (!err.ok()) return err;
    GgufValue value;
    err = read_value(&r, static_cast<GgufValueType>(vt), &value);
    if (!err.ok()) return err;
    file.kv.emplace(std::move(key), std::move(value));
  }

  file.tensors.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo t;
    err = r.read_string(&t.name);
    if (!err.ok()) return err;
    std::uint32_t n_dims = 0;
    err = r.read_pod(&n_dims);
    if (!err.ok()) return err;
    t.dims.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      err = r.read_pod(&t.dims[d]);
      if (!err.ok()) return err;
    }
    std::uint32_t typ = 0;
    err = r.read_pod(&typ);
    if (!err.ok()) return err;
    t.type = static_cast<GgufType>(typ);
    err = r.read_pod(&t.offset);
    if (!err.ok()) return err;
    file.tensors.push_back(std::move(t));
  }

  // Align to 32 bytes for data section.
  std::size_t pos = r.tell();
  const std::size_t align = 32;
  pos = (pos + align - 1) & ~(align - 1);
  file.data_offset = pos;
  *out = std::move(file);
  return Error::success();
}

Error gguf_load_tensor_f32(const GgufFile& file,
                           const std::string& tensor_name,
                           std::vector<float>* out,
                           Shape* out_shape) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "out null");
  }
  std::string key = tensor_name;
  const int expert = parse_expert_slice(&key);
  const GgufTensorInfo* info = nullptr;
  for (const auto& t : file.tensors) {
    if (t.name == key) {
      info = &t;
      break;
    }
  }
  if (info == nullptr) {
    return Error::make(ErrorCode::NotFound, "GGUF tensor not found: " + key);
  }

  const std::size_t n_full = numel_of(info->dims);
  std::size_t e_count = 1;
  std::size_t e_idx = 0;
  if (expert >= 0) {
    if (info->dims.size() < 3) {
      return Error::make(ErrorCode::InvalidArgument,
                         "GGUF expert slice needs rank-3 bank: " + key);
    }
    e_count = static_cast<std::size_t>(info->dims.back());
    if (e_count == 0 || n_full % e_count != 0 ||
        static_cast<std::size_t>(expert) >= e_count) {
      return Error::make(ErrorCode::InvalidArgument,
                         "GGUF expert slice out of range for " + key);
    }
    e_idx = static_cast<std::size_t>(expert);
  }
  const std::size_t n = n_full / e_count;
  const auto qf = gguf_type_to_quant(info->type);
  std::size_t packed_before = quant::packed_nbytes(qf, e_idx * n);
  if (packed_before == 0) {
    packed_before = e_idx * n * gguf_type_nbytes_per_elem(info->type);
  }

  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(file.path, &r);
  if (!err.ok()) return err;
  r.seek(static_cast<std::size_t>(file.data_offset + info->offset + packed_before));

  out->resize(n);
  if (info->type == GgufType::F32) {
    err = r.read_bytes(out->data(), n * 4);
    if (!err.ok()) return err;
  } else if (info->type == GgufType::F16) {
    std::vector<std::uint16_t> tmp(n);
    err = r.read_bytes(tmp.data(), n * 2);
    if (!err.ok()) return err;
    for (std::size_t i = 0; i < n; ++i) {
      (*out)[i] = f16_to_f32(tmp[i]);
    }
  } else if (info->type == GgufType::Q8_0) {
    // Block: f16 scale + 32 int8 values (34 bytes), QK=32
    constexpr std::size_t QK = 32;
    if (n % QK != 0) {
      return Error::make(ErrorCode::InvalidArgument, "Q8_0 numel not multiple of 32");
    }
    const std::size_t nblocks = n / QK;
    for (std::size_t b = 0; b < nblocks; ++b) {
      std::uint16_t dbits = 0;
      err = r.read_pod(&dbits);
      if (!err.ok()) return err;
      const float d = f16_to_f32(dbits);
      std::int8_t qs[QK];
      err = r.read_bytes(qs, QK);
      if (!err.ok()) return err;
      for (std::size_t i = 0; i < QK; ++i) {
        (*out)[b * QK + i] = d * static_cast<float>(qs[i]);
      }
    }
  } else if (info->type == GgufType::Q4_0) {
    // Block: f16 scale + 16 bytes nibbles (18 bytes), QK=32
    constexpr std::size_t QK = 32;
    if (n % QK != 0) {
      return Error::make(ErrorCode::InvalidArgument, "Q4_0 numel not multiple of 32");
    }
    const std::size_t nblocks = n / QK;
    for (std::size_t b = 0; b < nblocks; ++b) {
      std::uint16_t dbits = 0;
      err = r.read_pod(&dbits);
      if (!err.ok()) return err;
      const float d = f16_to_f32(dbits);
      std::uint8_t qs[16];
      err = r.read_bytes(qs, 16);
      if (!err.ok()) return err;
      for (std::size_t i = 0; i < QK; i += 2) {
        const std::uint8_t byte = qs[i / 2];
        const int x0 = static_cast<int>(byte & 0x0f) - 8;
        const int x1 = static_cast<int>(byte >> 4) - 8;
        (*out)[b * QK + i] = d * static_cast<float>(x0);
        (*out)[b * QK + i + 1] = d * static_cast<float>(x1);
      }
    }
  } else if (info->type == GgufType::I8) {
    std::vector<std::int8_t> tmp(n);
    err = r.read_bytes(tmp.data(), n);
    if (!err.ok()) return err;
    for (std::size_t i = 0; i < n; ++i) (*out)[i] = static_cast<float>(tmp[i]);
  } else if (info->type == GgufType::Q4_1 || info->type == GgufType::Q5_0 ||
             info->type == GgufType::Q5_1 || info->type == GgufType::Q2_K ||
             info->type == GgufType::Q3_K || info->type == GgufType::Q4_K ||
             info->type == GgufType::Q5_K || info->type == GgufType::Q6_K ||
             info->type == GgufType::IQ4_NL) {
    // K-quants / extra block quants: load packed bytes then dequant to f32.
    if (!quant::supports_gguf_dequant(qf)) {
      return Error::make(ErrorCode::NotImplemented,
                         "no dequant for GGUF dtype: " + std::string(to_string(info->type)));
    }
    std::vector<std::uint8_t> packed;
    Shape shape;
    GgufType ty = info->type;
    err = gguf_load_tensor_raw(file, tensor_name, &packed, &shape, &ty);
    if (!err.ok()) return err;
    err = quant::dequant_gguf_row(qf, packed.data(), static_cast<std::int64_t>(n),
                                  out->data());
    if (!err.ok()) return err;
  } else {
    return Error::make(ErrorCode::NotImplemented,
                       "unsupported GGUF dtype for f32 load: " +
                           std::string(to_string(info->type)) +
                           " (supported: F32/F16/Q4_0/Q4_1/Q5_*/Q8_0/Q*_K/IQ4_NL/I8)");
  }

  if (out_shape) {
    *out_shape = dims_to_shape(info->dims);
    if (expert >= 0 && !out_shape->dims.empty()) {
      out_shape->dims.erase(out_shape->dims.begin());
    }
  }
  return Error::success();
}

Error gguf_load_tensor_raw(const GgufFile& file,
                           const std::string& tensor_name,
                           std::vector<std::uint8_t>* out,
                           Shape* out_shape,
                           GgufType* out_type) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "raw out null");
  }
  std::string key = tensor_name;
  const int expert = parse_expert_slice(&key);
  const GgufTensorInfo* info = nullptr;
  for (const auto& t : file.tensors) {
    if (t.name == key) {
      info = &t;
      break;
    }
  }
  if (info == nullptr) {
    return Error::make(ErrorCode::NotFound, "tensor not in gguf: " + key);
  }
  const std::size_t n_full = numel_of(info->dims);
  std::size_t e_count = 1;
  std::size_t e_idx = 0;
  if (expert >= 0) {
    if (info->dims.size() < 3) {
      return Error::make(ErrorCode::InvalidArgument,
                         "GGUF expert slice needs rank-3 bank: " + key);
    }
    e_count = static_cast<std::size_t>(info->dims.back());
    if (e_count == 0 || n_full % e_count != 0 ||
        static_cast<std::size_t>(expert) >= e_count) {
      return Error::make(ErrorCode::InvalidArgument,
                         "GGUF expert slice out of range for " + key);
    }
    e_idx = static_cast<std::size_t>(expert);
  }
  const std::size_t n = n_full / e_count;
  if (!gguf_type_supported(info->type)) {
    return Error::make(ErrorCode::NotImplemented,
                       "unsupported GGUF type for raw load: " +
                           std::string(to_string(info->type)));
  }
  const auto qf = gguf_type_to_quant(info->type);
  std::size_t nbytes = quant::packed_nbytes(qf, n);
  if (nbytes == 0) {
    nbytes = n * gguf_type_nbytes_per_elem(info->type);
  }
  if (nbytes == 0) {
    return Error::make(ErrorCode::NotImplemented,
                       "unsupported GGUF type for raw load: " +
                           std::string(to_string(info->type)));
  }
  std::size_t packed_before = quant::packed_nbytes(qf, e_idx * n);
  if (packed_before == 0) {
    packed_before = e_idx * n * gguf_type_nbytes_per_elem(info->type);
  }
  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(file.path, &r);
  if (!err.ok()) return err;
  r.seek(static_cast<std::size_t>(file.data_offset + info->offset + packed_before));
  out->resize(nbytes);
  err = r.read_bytes(out->data(), nbytes);
  if (!err.ok()) return err;
  if (out_shape) {
    *out_shape = dims_to_shape(info->dims);
    if (expert >= 0 && !out_shape->dims.empty()) {
      out_shape->dims.erase(out_shape->dims.begin());
    }
  }
  if (out_type) *out_type = info->type;
  return Error::success();
}

Error gguf_write_f32(
    const std::string& path,
    const std::unordered_map<std::string, GgufValue>& kv,
    const std::vector<std::pair<GgufTensorInfo, std::vector<float>>>& tensors) {
  detail::ByteWriter w;
  const char magic[4] = {'G', 'G', 'U', 'F'};
  w.write_pod(magic, 4);
  w.write_pod(std::uint32_t{3});  // version
  w.write_pod(static_cast<std::uint64_t>(tensors.size()));
  w.write_pod(static_cast<std::uint64_t>(kv.size()));

  for (const auto& kv_pair : kv) {
    w.write_string(kv_pair.first);
    if (std::holds_alternative<std::string>(kv_pair.second)) {
      write_value(&w, GgufValueType::String, kv_pair.second);
    } else if (std::holds_alternative<std::uint32_t>(kv_pair.second)) {
      write_value(&w, GgufValueType::Uint32, kv_pair.second);
    } else if (std::holds_alternative<std::int32_t>(kv_pair.second)) {
      write_value(&w, GgufValueType::Int32, kv_pair.second);
    } else if (std::holds_alternative<float>(kv_pair.second)) {
      write_value(&w, GgufValueType::Float32, kv_pair.second);
    } else if (std::holds_alternative<bool>(kv_pair.second)) {
      write_value(&w, GgufValueType::Bool, kv_pair.second);
    } else if (std::holds_alternative<std::uint64_t>(kv_pair.second)) {
      write_value(&w, GgufValueType::Uint64, kv_pair.second);
    } else if (std::holds_alternative<std::vector<std::string>>(kv_pair.second)) {
      write_array(&w, GgufValueType::String, kv_pair.second);
    } else if (std::holds_alternative<std::vector<std::int32_t>>(kv_pair.second)) {
      write_array(&w, GgufValueType::Int32, kv_pair.second);
    } else if (std::holds_alternative<std::vector<std::int64_t>>(kv_pair.second)) {
      write_array(&w, GgufValueType::Int64, kv_pair.second);
    } else if (std::holds_alternative<std::vector<float>>(kv_pair.second)) {
      write_array(&w, GgufValueType::Float32, kv_pair.second);
    } else {
      write_value(&w, GgufValueType::String, GgufValue{std::string{}});
    }
  }

  std::uint64_t offset = 0;
  for (const auto& tp : tensors) {
    GgufTensorInfo info = tp.first;
    info.type = GgufType::F32;
    info.offset = offset;
    w.write_string(info.name);
    w.write_pod(static_cast<std::uint32_t>(info.dims.size()));
    for (auto d : info.dims) w.write_pod(d);
    w.write_pod(static_cast<std::uint32_t>(info.type));
    w.write_pod(info.offset);
    offset += static_cast<std::uint64_t>(tp.second.size() * sizeof(float));
  }

  w.pad_to(32);
  for (const auto& tp : tensors) {
    if (!tp.second.empty()) {
      w.write_pod(tp.second.data(), tp.second.size() * sizeof(float));
    }
  }
  return w.save(path);
}

LoaderInfo GgufLoader::info() const {
  return LoaderInfo{"gguf", "1.0", {".gguf"}};
}

bool GgufLoader::accepts(const std::string& path) const {
  return detail::lower_ext(path) == ".gguf";
}

namespace {

std::int64_t kv_i64(const GgufFile& file, const char* key, std::int64_t def) {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return def;
  const auto& v = it->second;
  if (std::holds_alternative<std::int64_t>(v)) return std::get<std::int64_t>(v);
  if (std::holds_alternative<std::uint64_t>(v))
    return static_cast<std::int64_t>(std::get<std::uint64_t>(v));
  if (std::holds_alternative<std::int32_t>(v)) return std::get<std::int32_t>(v);
  if (std::holds_alternative<std::uint32_t>(v))
    return static_cast<std::int64_t>(std::get<std::uint32_t>(v));
  return def;
}

std::string kv_string(const GgufFile& file, const char* key, const char* def = "") {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return def;
  const auto& v = it->second;
  if (std::holds_alternative<std::string>(v)) return std::get<std::string>(v);
  return def;
}

float kv_float(const GgufFile& file, const char* key, float def) {
  auto it = file.kv.find(key);
  if (it == file.kv.end()) return def;
  const auto& v = it->second;
  if (std::holds_alternative<float>(v)) return std::get<float>(v);
  if (std::holds_alternative<double>(v)) return static_cast<float>(std::get<double>(v));
  if (std::holds_alternative<std::int32_t>(v)) return static_cast<float>(std::get<std::int32_t>(v));
  if (std::holds_alternative<std::int64_t>(v)) return static_cast<float>(std::get<std::int64_t>(v));
  if (std::holds_alternative<std::uint32_t>(v))
    return static_cast<float>(std::get<std::uint32_t>(v));
  if (std::holds_alternative<std::uint64_t>(v))
    return static_cast<float>(std::get<std::uint64_t>(v));
  return def;
}

std::string lower_ascii(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

const GgufTensorInfo* find_tensor(const GgufFile& file, const std::string& name) {
  for (const auto& t : file.tensors) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

const GgufTensorInfo* find_tensor_any(const GgufFile& file,
                                      std::initializer_list<const char*> names) {
  for (const char* n : names) {
    if (auto* t = find_tensor(file, n)) return t;
  }
  return nullptr;
}

/// GGUF stores hyperparams as `{general.architecture}.key` (e.g. qwen2.block_count).
/// Probe arch prefix, then llama/general fallbacks used by many converters.
std::int64_t kv_i64_arch(const GgufFile& file, const std::string& arch, const char* suffix,
                         std::int64_t def) {
  if (!arch.empty()) {
    const std::string key = arch + "." + suffix;
    const std::int64_t v = kv_i64(file, key.c_str(), -1);
    if (v >= 0) return v;
  }
  const std::string llama_key = std::string("llama.") + suffix;
  const std::int64_t llama = kv_i64(file, llama_key.c_str(), -1);
  if (llama >= 0) return llama;
  const std::string general_key = std::string("general.") + suffix;
  const std::int64_t general = kv_i64(file, general_key.c_str(), -1);
  if (general >= 0) return general;
  return def;
}

float kv_float_arch(const GgufFile& file, const std::string& arch, const char* suffix, float def) {
  if (!arch.empty()) {
    const std::string key = arch + "." + suffix;
    auto it = file.kv.find(key);
    if (it != file.kv.end()) return kv_float(file, key.c_str(), def);
  }
  const std::string llama_key = std::string("llama.") + suffix;
  auto it_llama = file.kv.find(llama_key);
  if (it_llama != file.kv.end()) return kv_float(file, llama_key.c_str(), def);
  const std::string general_key = std::string("general.") + suffix;
  auto it_gen = file.kv.find(general_key);
  if (it_gen != file.kv.end()) return kv_float(file, general_key.c_str(), def);
  return def;
}

std::string kv_string_arch(const GgufFile& file, const std::string& arch, const char* suffix,
                           const char* def = "") {
  if (!arch.empty()) {
    const std::string key = arch + "." + suffix;
    auto it = file.kv.find(key);
    if (it != file.kv.end()) return kv_string(file, key.c_str(), def);
  }
  const std::string llama_key = std::string("llama.") + suffix;
  if (file.kv.find(llama_key) != file.kv.end()) return kv_string(file, llama_key.c_str(), def);
  return def;
}

bool looks_like_moe(const GgufFile& file, const std::string& arch) {
  if (kv_i64_arch(file, arch, "expert_count", 0) > 0) return true;
  static const char* kMoEHints[] = {
      "blk.0.ffn_gate_inp.weight",
      "blk.0.ffn_gate_exps.weight",
      "blk.0.ffn_up_exps.weight",
      "blk.0.ffn_down_exps.weight",
  };
  for (const char* n : kMoEHints) {
    if (find_tensor(file, n)) return true;
  }
  return false;
}

std::int64_t apply_layer_cap(std::int64_t model_layers) {
  constexpr std::int64_t kAbsoluteMax = 512;
  std::int64_t n = model_layers;
  if (n > kAbsoluteMax) {
    log::warn("gguf") << "layer count " << n << " exceeds absolute max " << kAbsoluteMax
                      << ", clamping";
    n = kAbsoluteMax;
  }
  const char* env = std::getenv("UAII_MAX_LAYERS");
  if (env != nullptr && env[0] != '\0') {
    const int cap = std::atoi(env);
    if (cap > 0 && n > cap) {
      log::warn("gguf") << "UAII_MAX_LAYERS=" << cap << " clamping layers from " << n;
      n = cap;
    }
    // cap == 0: unlimited (subject only to kAbsoluteMax above)
  }
  return n;
}

std::string detect_chat_template_meta(const GgufFile& file, const std::string& arch) {
  auto it = file.kv.find("tokenizer.chat_template");
  if (it != file.kv.end() && std::holds_alternative<std::string>(it->second)) {
    const std::string& j = std::get<std::string>(it->second);
    if (j.find("<|start_header_id|>") != std::string::npos) return "llama3";
    if (j.find("<start_of_turn>") != std::string::npos) return "gemma";
    if (j.find("[INST]") != std::string::npos) return "mistral";
    if (j.find("<|im_start|>") != std::string::npos) return "chatml";
  }
  if (arch == "llama" || arch == "llama3" || arch == "llama4") return "llama3";
  if (arch == "gemma" || arch == "gemma2" || arch == "gemma3") return "gemma";
  if (arch == "mistral" || arch == "mixtral" || arch == "llama2") return "mistral";
  return "chatml";
}

Error validate_gguf_model(const GgufFile& file, const std::string& arch) {
  static const char* kUnsupported[] = {
      "mamba",     "mamba2",    "rwkv",      "rwkv6",     "rwkv7",    "plamo2", "jamba",
      "falcon-h1", "bert",      "nomic-bert", "t5",       "t5encoder", "umt5",  "mt5",
      "flan-t5",   "clip",      "wav2vec2",  "whisper",
  };
  for (const char* u : kUnsupported) {
    if (arch == u) {
      std::string hint;
      if (arch == "mamba" || arch == "mamba2" || arch == "jamba" || arch == "falcon-h1" ||
          arch == "plamo2") {
        hint = "plugin ops SSMConv1d / SSMState (research_ops scaffold)";
      } else if (arch == "rwkv" || arch == "rwkv6" || arch == "rwkv7") {
        hint = "plugin op RWKVTimeMix (research_ops scaffold)";
      } else if (arch == "bert" || arch == "nomic-bert" || arch == "t5" || arch == "t5encoder" ||
                 arch == "umt5" || arch == "mt5" || arch == "flan-t5") {
        hint = "plugin op EncoderCrossAttn (research_ops scaffold)";
      } else if (arch == "clip") {
        hint = "plugin op VisionPatchEmbed (research_ops scaffold)";
      } else {
        hint = "SSM/encoder-decoder/vision kernels";
      }
      return Error::make(
          ErrorCode::NotImplemented,
          std::string("architecture '") + arch +
              "' is not supported yet (needs " + hint + "). "
              "Use a decoder-only Llama/Qwen/Gemma GGUF, or set UAII_MAX_LAYERS for partial tests.");
    }
  }
  if (find_tensor(file, "blk.0.ssm_conv1d.weight") ||
      find_tensor(file, "ssm_conv1d.weight")) {
    return Error::make(
        ErrorCode::NotImplemented,
        "Mamba/SSM models are not supported yet (needs plugin ops SSMConv1d / SSMState)");
  }
  if (find_tensor(file, "encoder.embd.weight") || find_tensor(file, "enc.embd.weight")) {
    return Error::make(
        ErrorCode::NotImplemented,
        "encoder-decoder models are not supported yet (needs plugin op EncoderCrossAttn)");
  }
  return Error::success();
}

/// Weight layout after dims_to_shape is [out, in] for MatMul(transpose_b=true).
std::int64_t weight_out_dim(const GgufTensorInfo* ti) {
  if (!ti || ti->dims.empty()) return 0;
  Shape s = dims_to_shape(ti->dims);
  return s.dims.empty() ? 0 : s.dims[0];
}

std::int64_t weight_in_dim(const GgufTensorInfo* ti) {
  if (!ti || ti->dims.size() < 2) return 0;
  Shape s = dims_to_shape(ti->dims);
  return s.dims.size() > 1 ? s.dims[1] : 0;
}

}  // namespace

Error GgufLoader::load(const std::string& path, ir::Graph* out_graph) {
  if (out_graph == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out null");
  }
  GgufFile file;
  Error err = gguf_read_header(path, &file);
  if (!err.ok()) return err;

  const GgufTensorInfo* emb =
      find_tensor_any(file, {"token_embd.weight", "embed.weight",
                             "model.embed_tokens.weight", "token_embd"});
  // Many GGUF models tie lm_head to token embeddings (no separate output.weight).
  const GgufTensorInfo* lm =
      find_tensor_any(file, {"output.weight", "lm_head.weight", "output"});
  const bool tied_embeddings = (lm == nullptr && emb != nullptr);
  if (tied_embeddings) lm = emb;
  const GgufTensorInfo* norm =
      find_tensor_any(file, {"output_norm.weight", "norm.weight", "output_norm"});

  const bool has_blk0 = find_tensor(file, "blk.0.attn_q.weight") != nullptr ||
                        find_tensor(file, "blk.0.attn_qkv.weight") != nullptr ||
                        find_tensor(file, "blk.0.attn_norm.weight") != nullptr ||
                        find_tensor(file, "blk.0.attn_kv_a_mqa.weight") != nullptr ||
                        find_tensor(file, "blk.0.attn_kv_a.weight") != nullptr ||
                        find_tensor(file, "blk.0.attn_q_a.weight") != nullptr;
  const std::string arch = lower_ascii(kv_string(file, "general.architecture"));

  err = validate_gguf_model(file, arch);
  if (!err.ok()) return err;

  // Capability-based import: any architecture with llama.cpp-style blk.* decoder
  // tensors is accepted (dense SwiGLU/GELU and Mixtral/Qwen2MoE-style expert FFN).
  ir::GraphBuilder b(emb && lm ? (has_blk0 ? "gguf_transformer" : "gguf_tiny_lm")
                               : "gguf_weights");
  b.set_producer("uaii-gguf-loader");
  b.set_metadata("source_format", "gguf");
  b.set_metadata("source_path", path);
  if (!arch.empty()) b.set_metadata("architecture", arch);
  if (tied_embeddings) b.set_metadata("tied_embeddings", "1");
  if (looks_like_moe(file, arch)) b.set_metadata("moe", "1");
  if (auto it = file.kv.find("general.name");
      it != file.kv.end() && std::holds_alternative<std::string>(it->second)) {
    b.set_metadata("model_name", std::get<std::string>(it->second));
  }

  if (emb && lm) {
    Shape emb_shape = dims_to_shape(emb->dims);
    Shape lm_shape = dims_to_shape(lm->dims);
    const std::int64_t dim = emb_shape.dims.size() > 1 ? emb_shape.dims[1] : 1;
    const std::int64_t vocab =
        lm_shape.dims.empty() ? emb_shape.dims[0] : lm_shape.dims[0];
    // Prefer arch-prefixed keys (qwen2.*, gemma.*, phi3.*, …) per GGUF spec.
    std::int64_t n_heads = kv_i64_arch(file, arch, "attention.head_count", 0);
    if (n_heads <= 0) n_heads = 1;
    std::int64_t n_kv_heads = kv_i64_arch(file, arch, "attention.head_count_kv", n_heads);
    if (n_kv_heads <= 0) n_kv_heads = n_heads;
    const std::int64_t emb_len = kv_i64_arch(file, arch, "embedding_length", dim);
    // Prefer explicit key/value length when present (GQA); else emb/n_heads.
    std::int64_t head_dim = kv_i64_arch(file, arch, "attention.key_length", 0);
    if (head_dim <= 0) {
      head_dim = n_heads > 0 ? (emb_len > 0 ? emb_len : dim) / n_heads : dim;
    }
    const std::int64_t q_dim = n_heads * head_dim;
    const std::int64_t kv_dim = n_kv_heads * head_dim;
    const float rope_theta = kv_float_arch(file, arch, "rope.freq_base", 10000.f);
    std::string rope_scale_type =
        lower_ascii(kv_string_arch(file, arch, "rope.scaling.type", ""));
    if (rope_scale_type.empty()) {
      rope_scale_type = lower_ascii(kv_string_arch(file, arch, "rope.scaling.rope_type", ""));
    }
    const float rope_scale_factor = kv_float_arch(file, arch, "rope.scaling.factor", 1.f);
    const float rope_low_f = kv_float_arch(file, arch, "rope.scaling.low_freq_factor", 1.f);
    const float rope_high_f = kv_float_arch(file, arch, "rope.scaling.high_freq_factor", 4.f);
    const float rope_old_ctx = kv_float_arch(
        file, arch, "rope.scaling.original_context_length",
        kv_float_arch(file, arch, "context_length", 8192.f));
    const float rope_attn_factor = kv_float_arch(file, arch, "rope.scaling.attn_factor", 1.f);
    const std::int64_t gguf_sliding =
        kv_i64_arch(file, arch, "attention.sliding_window",
                    kv_i64_arch(file, arch, "sliding_window", 0));
    // llama.cpp rope type: 0=NORMAL (interleaved), 2=NEOX (half pairing).
    // Qwen2/Phi/Falcon and many non-Llama GGUFs need NeoX.
    const int rope_mode =
        (arch == "qwen2" || arch == "qwen2moe" || arch == "qwen3" || arch == "qwen" ||
         arch == "phi2" || arch == "falcon" || arch == "stablelm" ||
         arch == "starcoder2" || arch == "olmo2" || arch == "gemma" || arch == "gemma2" ||
         arch == "gemma3" || arch == "deepseek2" || arch == "deepseekv2" ||
         arch == "deepseekv3" || arch == "deepseek")
            ? 2
            : 0;
    const std::int64_t q_lora_rank = kv_i64_arch(file, arch, "attention.q_lora_rank",
                                                 kv_i64_arch(file, arch, "q_lora_rank", 0));
    const std::int64_t kv_lora_rank = kv_i64_arch(file, arch, "attention.kv_lora_rank",
                                                  kv_i64_arch(file, arch, "kv_lora_rank", 0));
    const std::int64_t qk_nope_head_dim =
        kv_i64_arch(file, arch, "attention.qk_nope_head_dim",
                    kv_i64_arch(file, arch, "qk_nope_head_dim", 0));
    const std::int64_t qk_rope_head_dim =
        kv_i64_arch(file, arch, "attention.qk_rope_head_dim",
                    kv_i64_arch(file, arch, "qk_rope_head_dim", 0));
    const std::int64_t value_length = kv_i64_arch(file, arch, "attention.value_length", 0);
    // MLA RoPE chunk: prefer qk_rope_head_dim, else key_length, else dense head_dim.
    std::int64_t mla_rope_head_dim = qk_rope_head_dim;
    if (mla_rope_head_dim <= 0) {
      mla_rope_head_dim = kv_i64_arch(file, arch, "attention.key_length", 0);
    }
    if (mla_rope_head_dim <= 0) mla_rope_head_dim = head_dim;
    const float rms_eps =
        kv_float_arch(file, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
    const float layer_eps =
        kv_float_arch(file, arch, "attention.layer_norm_epsilon", rms_eps);
    const float attn_softcap = kv_float_arch(file, arch, "attn_logit_softcapping", 0.f);
    const bool use_layernorm =
        find_tensor(file, "blk.0.attn_norm.bias") != nullptr || arch == "gpt2" ||
        arch == "phi2";
    const std::string chat_tpl = detect_chat_template_meta(file, arch);
    const std::int64_t qkv_total = q_dim + kv_dim + kv_dim;
    std::int64_t n_layers = kv_i64_arch(file, arch, "block_count", 0);
    if (n_layers <= 0 && has_blk0) {
      // Infer layer count from tensor names.
      for (std::int64_t i = 0; i < 512; ++i) {
        const std::string probe = "blk." + std::to_string(i) + ".attn_norm.weight";
        if (!find_tensor(file, probe) &&
            !find_tensor(file, "blk." + std::to_string(i) + ".attn_q.weight") &&
            !find_tensor(file, "blk." + std::to_string(i) + ".attn_qkv.weight") &&
            !find_tensor(file, "blk." + std::to_string(i) + ".attn_kv_a_mqa.weight") &&
            !find_tensor(file, "blk." + std::to_string(i) + ".attn_kv_a.weight") &&
            !find_tensor(file, "blk." + std::to_string(i) + ".attn_q_a.weight")) {
          n_layers = i;
          break;
        }
      }
    }
    n_layers = apply_layer_cap(n_layers);
    const std::int64_t n_experts_meta = kv_i64_arch(file, arch, "expert_count", 0);
    const std::int64_t n_expert_used_meta =
        kv_i64_arch(file, arch, "expert_used_count", 0);
    if (n_experts_meta > 0) {
      b.set_metadata("expert_count", std::to_string(n_experts_meta));
    }
    if (n_expert_used_meta > 0) {
      b.set_metadata("expert_used_count", std::to_string(n_expert_used_meta));
    }
    if (q_lora_rank > 0) b.set_metadata("q_lora_rank", std::to_string(q_lora_rank));
    if (kv_lora_rank > 0) b.set_metadata("kv_lora_rank", std::to_string(kv_lora_rank));
    if (qk_nope_head_dim > 0) {
      b.set_metadata("qk_nope_head_dim", std::to_string(qk_nope_head_dim));
    }
    if (qk_rope_head_dim > 0) {
      b.set_metadata("qk_rope_head_dim", std::to_string(qk_rope_head_dim));
    }
    if (value_length > 0) b.set_metadata("value_length", std::to_string(value_length));
    bool any_mla = false;
    if (has_blk0) {
      log::info("gguf") << "transformer import arch="
                        << (arch.empty() ? "unspecified" : arch)
                        << " layers=" << n_layers << " heads=" << n_heads
                        << " kv_heads=" << n_kv_heads
                        << (n_experts_meta > 0
                                ? (" moe_experts=" + std::to_string(n_experts_meta) +
                                   " top_k=" +
                                   std::to_string(n_expert_used_meta > 0 ? n_expert_used_meta
                                                                       : 2))
                                : "");
    }

    // seq=1 decode path: tokens [1,1], activations [1,dim] (Embedding + Session::generate).
    TensorId tokens = b.add_tensor("tokens", DType::F32, Shape{{1, 1}});
    if (!gguf_type_supported(emb->type)) {
      return Error::make(ErrorCode::NotImplemented,
                         "unsupported GGUF quant type for weight " + emb->name + ": " +
                             to_string(emb->type));
    }
    TensorId emb_w =
        b.add_weight(emb->name, DType::F32, emb_shape, path + "#" + emb->name,
                     gguf_type_to_quant(emb->type));
    TensorId hidden = b.add_tensor("hidden", DType::F32, Shape{{1, dim}});
    b.add_node("embed", "Embedding", "1", {tokens, emb_w}, {hidden});

    TensorId features = hidden;
    auto add_weight_ref = [&](const std::string& name, TensorId* out_id) -> Error {
      const GgufTensorInfo* ti = find_tensor(file, name);
      if (!ti) {
        *out_id = 0;
        return Error::success();
      }
      if (!gguf_type_supported(ti->type)) {
        std::string hint =
            " Supported quants: f16,f32,q4_0,q4_1,q5_0,q5_1,q8_0,q2_k..q6_k,iq4_nl.";
        switch (ti->type) {
          case GgufType::IQ2_XXS:
          case GgufType::IQ2_XS:
          case GgufType::IQ3_XXS:
          case GgufType::IQ1_S:
          case GgufType::IQ3_S:
          case GgufType::IQ2_S:
          case GgufType::IQ4_XS:
          case GgufType::IQ1_M:
          case GgufType::Q8_K:
            hint +=
                " IQ* / Q8_K types are not implemented — convert to Q4_K_M (or F16) with "
                "llama-quantize.";
            break;
          default:
            break;
        }
        return Error::make(ErrorCode::NotImplemented,
                           "unsupported GGUF quant type for weight " + name + ": " +
                               to_string(ti->type) + "." + hint);
      }
      *out_id = b.add_weight(name, DType::F32, dims_to_shape(ti->dims), path + "#" + name,
                             gguf_type_to_quant(ti->type));
      return Error::success();
    };

    for (std::int64_t li = 0; li < n_layers; ++li) {
      const std::string pfx = "blk." + std::to_string(li) + ".";
      TensorId attn_norm = 0;
      TensorId wq = 0;
      TensorId wk = 0;
      TensorId wv = 0;
      TensorId wo = 0;
      TensorId ffn_norm = 0;
      TensorId w_gate = 0;
      TensorId w_up = 0;
      TensorId w_down = 0;
      TensorId w_gate_inp = 0;
      TensorId w_gate_shexp = 0;
      TensorId w_up_shexp = 0;
      TensorId w_down_shexp = 0;
      TensorId w_q_norm = 0;
      TensorId w_k_norm = 0;
      TensorId w_qkv = 0;
      TensorId b_qkv = 0;
      TensorId attn_norm_b = 0;
      TensorId ffn_norm_b = 0;
      err = add_weight_ref(pfx + "attn_norm.weight", &attn_norm);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_norm.bias", &attn_norm_b);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_qkv.weight", &w_qkv);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_q.weight", &wq);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_k.weight", &wk);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_v.weight", &wv);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_q_norm.weight", &w_q_norm);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_k_norm.weight", &w_k_norm);
      if (!err.ok()) return err;
      // DeepSeek MLA (expand-then-attend): optional Q-LoRA + KV absorb/expand.
      TensorId w_q_a = 0;
      TensorId w_q_a_norm = 0;
      TensorId w_q_b = 0;
      TensorId w_kv_a = 0;
      TensorId w_kv_a_norm = 0;
      TensorId w_k_b = 0;
      TensorId w_v_b = 0;
      err = add_weight_ref(pfx + "attn_q_a.weight", &w_q_a);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_q_a_norm.weight", &w_q_a_norm);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_q_b.weight", &w_q_b);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_kv_a_mqa.weight", &w_kv_a);
      if (!err.ok()) return err;
      if (!w_kv_a) {
        err = add_weight_ref(pfx + "attn_kv_a.weight", &w_kv_a);
        if (!err.ok()) return err;
      }
      err = add_weight_ref(pfx + "attn_kv_a_norm.weight", &w_kv_a_norm);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_k_b.weight", &w_k_b);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_v_b.weight", &w_v_b);
      if (!err.ok()) return err;
      TensorId bq = 0;
      TensorId bk = 0;
      TensorId bv = 0;
      err = add_weight_ref(pfx + "attn_q.bias", &bq);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_k.bias", &bk);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_v.bias", &bv);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_qkv.bias", &b_qkv);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_output.weight", &wo);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_norm.weight", &ffn_norm);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_norm.bias", &ffn_norm_b);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_gate.weight", &w_gate);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_up.weight", &w_up);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_down.weight", &w_down);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_gate_inp.weight", &w_gate_inp);
      if (!err.ok()) return err;
      // Packed Mixtral/Qwen banks are split into per-expert IR tensors below
      // (`#e=N` slices). Do not register the whole bank as one weight.
      err = add_weight_ref(pfx + "ffn_gate_shexp.weight", &w_gate_shexp);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_up_shexp.weight", &w_up_shexp);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_down_shexp.weight", &w_down_shexp);
      if (!err.ok()) return err;

      const bool has_kv_a = (find_tensor(file, pfx + "attn_kv_a_mqa.weight") != nullptr) ||
                            (find_tensor(file, pfx + "attn_kv_a.weight") != nullptr);
      const bool is_mla = has_kv_a && w_k_b != 0 && w_v_b != 0;
      const bool split_qkv = (wq != 0 && wk != 0 && wv != 0);
      const bool fused_qkv = (w_qkv != 0);
      if (!attn_norm || !wo) {
        break;
      }
      if (has_kv_a || w_k_b || w_v_b || w_q_a || w_q_b) {
        if (!is_mla || !w_kv_a) {
          return Error::make(
              ErrorCode::InvalidArgument,
              pfx + " incomplete MLA tensors (need attn_kv_a(_mqa)+attn_k_b+attn_v_b, "
                    "and attn_q_a+attn_q_b or attn_q)");
        }
        if (!((w_q_a && w_q_b) || wq)) {
          return Error::make(ErrorCode::InvalidArgument,
                             pfx + " MLA incomplete: need attn_q_a+attn_q_b or attn_q");
        }
        any_mla = true;
      } else if (!split_qkv && !fused_qkv) {
        break;
      }

      TensorId n1 = b.add_tensor(pfx + "n1", DType::F32, Shape{{1, dim}});
      if (use_layernorm && attn_norm_b) {
        b.add_node(pfx + "ln1", "LayerNorm", "1", {features, attn_norm, attn_norm_b}, {n1},
                   {ir::make_float_attr("eps", layer_eps)});
      } else {
        b.add_node(pfx + "rms1", "RMSNorm", "1", {features, attn_norm}, {n1},
                   {ir::make_float_attr("eps", rms_eps)});
      }

      TensorId q = 0;
      TensorId k = 0;
      TensorId v = 0;
      std::int64_t layer_q_dim = q_dim;
      std::int64_t layer_kv_dim = kv_dim;
      std::int64_t layer_v_dim = kv_dim;
      std::int64_t layer_heads = n_heads;
      std::int64_t layer_kv_heads = n_kv_heads;
      std::int64_t layer_rope_hd = head_dim;
      std::int64_t layer_attn_out = q_dim;
      bool mla_k_assembled = false;

      if (is_mla) {
        const GgufTensorInfo* ti_q_b = find_tensor(file, pfx + "attn_q_b.weight");
        const GgufTensorInfo* ti_q = find_tensor(file, pfx + "attn_q.weight");
        const GgufTensorInfo* ti_kv_a =
            find_tensor(file, pfx + "attn_kv_a_mqa.weight")
                ? find_tensor(file, pfx + "attn_kv_a_mqa.weight")
                : find_tensor(file, pfx + "attn_kv_a.weight");
        const GgufTensorInfo* ti_k_b = find_tensor(file, pfx + "attn_k_b.weight");
        const GgufTensorInfo* ti_v_b = find_tensor(file, pfx + "attn_v_b.weight");
        const GgufTensorInfo* ti_kv_a_norm = find_tensor(file, pfx + "attn_kv_a_norm.weight");

        layer_q_dim = w_q_a && w_q_b ? weight_out_dim(ti_q_b) : weight_out_dim(ti_q);
        layer_kv_dim = weight_out_dim(ti_k_b);
        layer_v_dim = weight_out_dim(ti_v_b);
        if (layer_q_dim <= 0 || layer_kv_dim <= 0 || layer_v_dim <= 0) {
          return Error::make(ErrorCode::InvalidArgument,
                             pfx + " MLA weight shapes invalid (q/k_b/v_b out dims)");
        }

        // After expand, MLA is typically MHA (kv_heads == n_heads).
        layer_heads = n_heads;
        layer_kv_heads = n_heads;
        layer_rope_hd = mla_rope_head_dim > 0 ? mla_rope_head_dim : head_dim;
        // Prefer full qk head (nope+rope) when metadata present so RoPE/Attention align.
        if (qk_nope_head_dim > 0 && qk_rope_head_dim > 0) {
          layer_rope_hd = qk_nope_head_dim + qk_rope_head_dim;
        } else if (layer_heads > 0 && layer_q_dim % layer_heads == 0) {
          layer_rope_hd = layer_q_dim / layer_heads;
        }
        layer_attn_out = layer_v_dim;

        // --- Q (optional Q-LoRA) ---
        if (w_q_a && w_q_b) {
          const std::int64_t q_a_dim = weight_out_dim(find_tensor(file, pfx + "attn_q_a.weight"));
          if (q_a_dim <= 0) {
            return Error::make(ErrorCode::InvalidArgument, pfx + " attn_q_a shape invalid");
          }
          TensorId qa = b.add_tensor(pfx + "q_a", DType::F32, Shape{{1, q_a_dim}});
          b.add_node(pfx + "q_a_mm", "MatMul", "1", {n1, w_q_a}, {qa},
                     {ir::make_bool_attr("transpose_b", true)});
          if (w_q_a_norm) {
            TensorId qa_n = b.add_tensor(pfx + "q_a_n", DType::F32, Shape{{1, q_a_dim}});
            b.add_node(pfx + "q_a_rms", "RMSNorm", "1", {qa, w_q_a_norm}, {qa_n},
                       {ir::make_float_attr("eps", rms_eps)});
            qa = qa_n;
          }
          q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, layer_q_dim}});
          b.add_node(pfx + "q_b_mm", "MatMul", "1", {qa, w_q_b}, {q},
                     {ir::make_bool_attr("transpose_b", true)});
        } else if (wq) {
          q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, layer_q_dim}});
          b.add_node(pfx + "q_mm", "MatMul", "1", {n1, wq}, {q},
                     {ir::make_bool_attr("transpose_b", true)});
        } else {
          return Error::make(ErrorCode::InvalidArgument, pfx + " MLA missing Q projection");
        }

        // --- KV absorb + expand (full K/V each step → existing Attention + KV cache) ---
        const std::int64_t kvc_full_dim = weight_out_dim(ti_kv_a);
        const std::int64_t k_b_in = weight_in_dim(ti_k_b);
        const std::int64_t v_b_in = weight_in_dim(ti_v_b);
        if (kvc_full_dim <= 0 || k_b_in <= 0 || v_b_in <= 0 || k_b_in != v_b_in) {
          return Error::make(ErrorCode::InvalidArgument,
                             pfx + " MLA kv_a/k_b/v_b in/out dims inconsistent");
        }
        TensorId kvc = b.add_tensor(pfx + "kvc", DType::F32, Shape{{1, kvc_full_dim}});
        b.add_node(pfx + "kv_a_mm", "MatMul", "1", {n1, w_kv_a}, {kvc},
                   {ir::make_bool_attr("transpose_b", true)});

        // attn_kv_a_mqa often emits [kv_lora_rank + qk_rope]; norm/k_b use kv_lora only.
        TensorId latent = kvc;
        std::int64_t latent_dim = kvc_full_dim;
        if (kvc_full_dim != k_b_in) {
          if (k_b_in > kvc_full_dim) {
            return Error::make(ErrorCode::InvalidArgument,
                               pfx + " MLA kv_a out < k_b in (incomplete MLA tensors)");
          }
          latent_dim = k_b_in;
          latent = b.add_tensor(pfx + "kv_latent", DType::F32, Shape{{1, latent_dim}});
          b.add_node(pfx + "kv_latent_slice", "Slice", "1", {kvc}, {latent},
                     {ir::make_int_attr("start", 0), ir::make_int_attr("end", latent_dim)});
        }
        if (w_kv_a_norm) {
          const std::int64_t norm_dim = weight_out_dim(ti_kv_a_norm);
          if (norm_dim > 0 && norm_dim != latent_dim) {
            return Error::make(ErrorCode::InvalidArgument,
                               pfx + " MLA attn_kv_a_norm size mismatch vs latent");
          }
          TensorId lat_n = b.add_tensor(pfx + "kv_latent_n", DType::F32, Shape{{1, latent_dim}});
          b.add_node(pfx + "kv_a_rms", "RMSNorm", "1", {latent, w_kv_a_norm}, {lat_n},
                     {ir::make_float_attr("eps", rms_eps)});
          latent = lat_n;
        }

        k = b.add_tensor(pfx + "k", DType::F32, Shape{{1, layer_kv_dim}});
        v = b.add_tensor(pfx + "v", DType::F32, Shape{{1, layer_v_dim}});
        b.add_node(pfx + "k_b_mm", "MatMul", "1", {latent, w_k_b}, {k},
                   {ir::make_bool_attr("transpose_b", true)});
        b.add_node(pfx + "v_b_mm", "MatMul", "1", {latent, w_v_b}, {v},
                   {ir::make_bool_attr("transpose_b", true)});

        // If K is nope-only and Q is nope+rope, merge shared RoPE K onto every head.
        if (layer_q_dim != layer_kv_dim) {
          const std::int64_t rope_extra = kvc_full_dim - latent_dim;
          std::int64_t nope = qk_nope_head_dim;
          std::int64_t rope = qk_rope_head_dim > 0 ? qk_rope_head_dim : rope_extra;
          if (nope <= 0 && layer_heads > 0 && layer_kv_dim % layer_heads == 0) {
            nope = layer_kv_dim / layer_heads;
          }
          if (rope <= 0 && layer_heads > 0 && layer_q_dim % layer_heads == 0 && nope > 0) {
            rope = (layer_q_dim / layer_heads) - nope;
          }
          if (nope <= 0 || rope <= 0 || rope_extra < rope ||
              layer_kv_dim != layer_heads * nope ||
              layer_q_dim != layer_heads * (nope + rope)) {
            return Error::make(
                ErrorCode::InvalidArgument,
                pfx + " MLA expand produced q_dim=" + std::to_string(layer_q_dim) +
                    " k_dim=" + std::to_string(layer_kv_dim) +
                    "; cannot assemble nope+rope K (check qk_nope/qk_rope metadata)");
          }
          TensorId k_pe = b.add_tensor(pfx + "k_pe", DType::F32, Shape{{1, rope}});
          b.add_node(pfx + "k_pe_slice", "Slice", "1", {kvc}, {k_pe},
                     {ir::make_int_attr("start", latent_dim),
                      ir::make_int_attr("end", latent_dim + rope)});
          TensorId k_pe_r = b.add_tensor(pfx + "k_pe_rope", DType::F32, Shape{{1, rope}});
          b.add_node(pfx + "k_pe_rope", "RoPE", "1", {k_pe}, {k_pe_r},
                     {ir::make_float_attr("theta", rope_theta),
                      ir::make_int_attr("head_dim", rope),
                      ir::make_int_attr("mode", rope_mode)});
          TensorId k_full =
              b.add_tensor(pfx + "k_full", DType::F32, Shape{{1, layer_q_dim}});
          b.add_node(pfx + "mla_expand_k", "MLAExpandK", "1", {k, k_pe_r}, {k_full},
                     {ir::make_int_attr("num_heads", layer_heads),
                      ir::make_int_attr("nope_dim", nope),
                      ir::make_int_attr("rope_dim", rope)});
          k = k_full;
          layer_kv_dim = layer_q_dim;
          layer_rope_hd = nope + rope;
          mla_k_assembled = true;
        }
      } else if (fused_qkv) {
        TensorId qkv_proj = b.add_tensor(pfx + "qkv_proj", DType::F32, Shape{{1, qkv_total}});
        b.add_node(pfx + "qkv_mm", "MatMul", "1", {n1, w_qkv}, {qkv_proj},
                   {ir::make_bool_attr("transpose_b", true)});
        TensorId qkv = qkv_proj;
        if (b_qkv) {
          qkv = b.add_tensor(pfx + "qkv", DType::F32, Shape{{1, qkv_total}});
          b.add_node(pfx + "qkv_bias", "Add", "1", {qkv_proj, b_qkv}, {qkv});
        }
        q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, q_dim}});
        k = b.add_tensor(pfx + "k", DType::F32, Shape{{1, kv_dim}});
        v = b.add_tensor(pfx + "v", DType::F32, Shape{{1, kv_dim}});
        b.add_node(pfx + "q_slice", "Slice", "1", {qkv}, {q},
                   {ir::make_int_attr("start", 0), ir::make_int_attr("end", q_dim)});
        b.add_node(pfx + "k_slice", "Slice", "1", {qkv}, {k},
                   {ir::make_int_attr("start", q_dim),
                    ir::make_int_attr("end", q_dim + kv_dim)});
        b.add_node(pfx + "v_slice", "Slice", "1", {qkv}, {v},
                   {ir::make_int_attr("start", q_dim + kv_dim),
                    ir::make_int_attr("end", qkv_total)});
      } else {
        TensorId q_proj = b.add_tensor(pfx + "q_proj", DType::F32, Shape{{1, q_dim}});
        TensorId k_proj = b.add_tensor(pfx + "k_proj", DType::F32, Shape{{1, kv_dim}});
        TensorId v_proj = b.add_tensor(pfx + "v_proj", DType::F32, Shape{{1, kv_dim}});
        b.add_node(pfx + "q_mm", "MatMul", "1", {n1, wq}, {q_proj},
                   {ir::make_bool_attr("transpose_b", true)});
        b.add_node(pfx + "k_mm", "MatMul", "1", {n1, wk}, {k_proj},
                   {ir::make_bool_attr("transpose_b", true)});
        b.add_node(pfx + "v_mm", "MatMul", "1", {n1, wv}, {v_proj},
                   {ir::make_bool_attr("transpose_b", true)});

        q = q_proj;
        k = k_proj;
        v = v_proj;
        if (bq) {
          q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, q_dim}});
          b.add_node(pfx + "q_bias", "Add", "1", {q_proj, bq}, {q});
        }
        if (bk) {
          k = b.add_tensor(pfx + "k", DType::F32, Shape{{1, kv_dim}});
          b.add_node(pfx + "k_bias", "Add", "1", {k_proj, bk}, {k});
        }
        if (bv) {
          v = b.add_tensor(pfx + "v", DType::F32, Shape{{1, kv_dim}});
          b.add_node(pfx + "v_bias", "Add", "1", {v_proj, bv}, {v});
        }
      }

      if (w_q_norm) {
        TensorId qn = b.add_tensor(pfx + "q_n", DType::F32, Shape{{1, layer_q_dim}});
        b.add_node(pfx + "q_norm", "RMSNorm", "1", {q, w_q_norm}, {qn},
                   {ir::make_float_attr("eps", rms_eps)});
        q = qn;
      }
      if (w_k_norm) {
        TensorId kn = b.add_tensor(pfx + "k_n", DType::F32, Shape{{1, layer_kv_dim}});
        b.add_node(pfx + "k_norm", "RMSNorm", "1", {k, w_k_norm}, {kn},
                   {ir::make_float_attr("eps", rms_eps)});
        k = kn;
      }

      auto add_gguf_rope_scale = [&](std::vector<ir::Attribute>* attrs) {
        if (rope_scale_type.empty() || rope_scale_type == "none" || rope_scale_type == "default") {
          return;
        }
        attrs->push_back(ir::make_string_attr("scale_type", rope_scale_type));
        attrs->push_back(ir::make_float_attr("scale_factor", rope_scale_factor));
        attrs->push_back(ir::make_float_attr("low_freq_factor", rope_low_f));
        attrs->push_back(ir::make_float_attr("high_freq_factor", rope_high_f));
        attrs->push_back(ir::make_float_attr("old_context_len", rope_old_ctx > 0.f ? rope_old_ctx : 8192.f));
        attrs->push_back(ir::make_float_attr("attn_factor", rope_attn_factor));
      };

      TensorId q_rope = b.add_tensor(pfx + "q_rope", DType::F32, Shape{{1, layer_q_dim}});
      TensorId k_rope = b.add_tensor(pfx + "k_rope", DType::F32, Shape{{1, layer_kv_dim}});
      {
        std::vector<ir::Attribute> q_rope_attrs{
            ir::make_float_attr("theta", rope_theta),
            ir::make_int_attr("head_dim", layer_rope_hd),
            ir::make_int_attr("mode", rope_mode)};
        if (is_mla && qk_rope_head_dim > 0 && layer_rope_hd > qk_rope_head_dim) {
          q_rope_attrs.push_back(ir::make_int_attr("rotary_dim", qk_rope_head_dim));
        }
        add_gguf_rope_scale(&q_rope_attrs);
        b.add_node(pfx + "q_rope", "RoPE", "1", {q}, {q_rope}, q_rope_attrs);
      }
      if (mla_k_assembled) {
        // Shared K rope already applied inside MLAExpandK inputs.
        b.add_node(pfx + "k_rope", "Identity", "1", {k}, {k_rope});
      } else {
        std::vector<ir::Attribute> k_rope_attrs{
            ir::make_float_attr("theta", rope_theta),
            ir::make_int_attr("head_dim", layer_rope_hd),
            ir::make_int_attr("mode", rope_mode)};
        if (is_mla && qk_rope_head_dim > 0 && layer_rope_hd > qk_rope_head_dim) {
          k_rope_attrs.push_back(ir::make_int_attr("rotary_dim", qk_rope_head_dim));
        }
        add_gguf_rope_scale(&k_rope_attrs);
        b.add_node(pfx + "k_rope", "RoPE", "1", {k}, {k_rope}, k_rope_attrs);
      }

      TensorId attn = b.add_tensor(pfx + "attn", DType::F32, Shape{{1, layer_attn_out}});
      {
        std::vector<ir::Attribute> attn_attrs{
            ir::make_int_attr("num_heads", layer_heads),
            ir::make_int_attr("kv_heads", layer_kv_heads),
            ir::make_bool_attr("causal", true),
            ir::make_bool_attr("use_kv_cache", true),
            ir::make_int_attr("layer_id", li)};
        if (attn_softcap > 0.f) {
          attn_attrs.push_back(ir::make_float_attr("logit_softcap", attn_softcap));
        }
        if (gguf_sliding > 0) {
          attn_attrs.push_back(ir::make_int_attr("sliding_window", gguf_sliding));
        }
        b.add_node(pfx + "attn", "Attention", "1", {q_rope, k_rope, v}, {attn}, attn_attrs);
      }

      TensorId attn_o = b.add_tensor(pfx + "attn_o", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "o_proj", "MatMul", "1", {attn, wo}, {attn_o},
                 {ir::make_bool_attr("transpose_b", true)});

      TensorId resid1 = b.add_tensor(pfx + "resid1", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "add1", "Add", "1", {features, attn_o}, {resid1});

      TensorId mlp_in = resid1;
      if (ffn_norm) {
        TensorId n2 = b.add_tensor(pfx + "n2", DType::F32, Shape{{1, dim}});
        if (use_layernorm && ffn_norm_b) {
          b.add_node(pfx + "ln2", "LayerNorm", "1", {resid1, ffn_norm, ffn_norm_b}, {n2},
                     {ir::make_float_attr("eps", layer_eps)});
        } else {
          b.add_node(pfx + "rms2", "RMSNorm", "1", {resid1, ffn_norm}, {n2},
                     {ir::make_float_attr("eps", rms_eps)});
        }
        mlp_in = n2;
      }

      auto inter_dim_from_up = [&](const GgufTensorInfo* up_info) -> std::int64_t {
        if (!up_info) return dim * 4;
        Shape up_shape = dims_to_shape(up_info->dims);
        if (up_shape.dims.size() == 2) {
          return up_shape.dims[0] == dim ? up_shape.dims[1] : up_shape.dims[0];
        }
        if (up_shape.dims.size() == 3) {
          // Expert stack [E, I, D] after dims_to_shape.
          return up_shape.dims[1];
        }
        return dim * 4;
      };

      if (w_gate_inp && find_tensor(file, pfx + "ffn_gate_exps.weight") &&
          find_tensor(file, pfx + "ffn_up_exps.weight") &&
          find_tensor(file, pfx + "ffn_down_exps.weight")) {
        // Mixtral / Qwen2MoE-style: router + top-k SwiGLU experts (+ optional shared).
        // Split packed [E,I,D]/[E,D,I] banks into listed per-expert tensors so the
        // expert LRU pages one expert (kimi-style) instead of the whole bank.
        const GgufTensorInfo* gate_info = find_tensor(file, pfx + "ffn_gate_exps.weight");
        const GgufTensorInfo* up_info = find_tensor(file, pfx + "ffn_up_exps.weight");
        const GgufTensorInfo* down_info = find_tensor(file, pfx + "ffn_down_exps.weight");
        Shape gate_shape = dims_to_shape(gate_info->dims);
        Shape up_shape = dims_to_shape(up_info->dims);
        Shape down_shape = dims_to_shape(down_info->dims);
        std::int64_t n_experts = kv_i64_arch(file, arch, "expert_count", 0);
        if (n_experts <= 0 && gate_shape.dims.size() >= 1) n_experts = gate_shape.dims[0];
        if (n_experts <= 0) {
          return Error::make(ErrorCode::InvalidArgument,
                             "MoE layer " + pfx + " missing expert_count / expert dims");
        }
        std::int64_t top_k = kv_i64_arch(file, arch, "expert_used_count", 2);
        if (top_k <= 0) top_k = 2;
        if (top_k > n_experts) top_k = n_experts;
        if (gate_shape.dims.size() != 3 || up_shape.dims.size() != 3 ||
            down_shape.dims.size() != 3 || gate_shape.dims[0] != n_experts ||
            up_shape.dims[0] != n_experts || down_shape.dims[0] != n_experts) {
          return Error::make(ErrorCode::InvalidArgument,
                             pfx + " packed expert banks must be rank-3 [E,...]");
        }
        const Shape g_one{{gate_shape.dims[1], gate_shape.dims[2]}};
        const Shape u_one{{up_shape.dims[1], up_shape.dims[2]}};
        const Shape d_one{{down_shape.dims[1], down_shape.dims[2]}};

        TensorId router_probs =
            b.add_tensor(pfx + "moe_probs", DType::F32, Shape{{1, n_experts}});
        TensorId router_top =
            b.add_tensor(pfx + "moe_top", DType::F32, Shape{{1, 1}});
        b.add_node(pfx + "moe_router", "MoERouter", "1", {mlp_in, w_gate_inp},
                   {router_probs, router_top},
                   {ir::make_int_attr("num_experts", n_experts),
                    ir::make_int_attr("top_k", top_k)});

        std::vector<TensorId> moe_ins = {mlp_in, router_probs};
        for (std::int64_t e = 0; e < n_experts; ++e) {
          const std::string es = std::to_string(e);
          moe_ins.push_back(b.add_weight(
              pfx + "exp." + es + ".ffn_gate.weight", DType::F32, g_one,
              path + "#" + gate_info->name + "#e=" + es,
              gguf_type_to_quant(gate_info->type)));
          moe_ins.push_back(b.add_weight(
              pfx + "exp." + es + ".ffn_up.weight", DType::F32, u_one,
              path + "#" + up_info->name + "#e=" + es, gguf_type_to_quant(up_info->type)));
          moe_ins.push_back(b.add_weight(
              pfx + "exp." + es + ".ffn_down.weight", DType::F32, d_one,
              path + "#" + down_info->name + "#e=" + es,
              gguf_type_to_quant(down_info->type)));
        }

        TensorId ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "moe_exps", "MoEExpertsSwiGLU", "1", moe_ins, {ff_out},
                   {ir::make_int_attr("num_experts", n_experts),
                    ir::make_int_attr("top_k", top_k)});

        if (w_gate_shexp && w_up_shexp && w_down_shexp) {
          const GgufTensorInfo* shexp_up = find_tensor(file, pfx + "ffn_up_shexp.weight");
          const std::int64_t shexp_inter = inter_dim_from_up(shexp_up);
          TensorId sg = b.add_tensor(pfx + "shexp_gate", DType::F32, Shape{{1, shexp_inter}});
          TensorId su = b.add_tensor(pfx + "shexp_up", DType::F32, Shape{{1, shexp_inter}});
          b.add_node(pfx + "shexp_gate_mm", "MatMul", "1", {mlp_in, w_gate_shexp}, {sg},
                     {ir::make_bool_attr("transpose_b", true)});
          b.add_node(pfx + "shexp_up_mm", "MatMul", "1", {mlp_in, w_up_shexp}, {su},
                     {ir::make_bool_attr("transpose_b", true)});
          TensorId sga =
              b.add_tensor(pfx + "shexp_gate_act", DType::F32, Shape{{1, shexp_inter}});
          b.add_node(pfx + "shexp_silu", "Silu", "1", {sg}, {sga});
          TensorId shid =
              b.add_tensor(pfx + "shexp_hid", DType::F32, Shape{{1, shexp_inter}});
          b.add_node(pfx + "shexp_mul", "Mul", "1", {sga, su}, {shid});
          TensorId sout = b.add_tensor(pfx + "shexp_out", DType::F32, Shape{{1, dim}});
          b.add_node(pfx + "shexp_down", "MatMul", "1", {shid, w_down_shexp}, {sout},
                     {ir::make_bool_attr("transpose_b", true)});
          TensorId merged = b.add_tensor(pfx + "ff_merged", DType::F32, Shape{{1, dim}});
          b.add_node(pfx + "ff_add_shexp", "Add", "1", {ff_out, sout}, {merged});
          ff_out = merged;
        }

        TensorId resid2 = b.add_tensor(pfx + "resid2", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "add2", "Add", "1", {resid1, ff_out}, {resid2});
        features = resid2;
      } else if (w_gate && w_up && w_down) {
        // SwiGLU-style: silu(x@gate) * (x@up) @ down
        const GgufTensorInfo* up_info = find_tensor(file, pfx + "ffn_up.weight");
        const std::int64_t inter_dim = inter_dim_from_up(up_info);

        TensorId gate = b.add_tensor(pfx + "gate", DType::F32, Shape{{1, inter_dim}});
        TensorId up = b.add_tensor(pfx + "up", DType::F32, Shape{{1, inter_dim}});
        b.add_node(pfx + "ffn_gate", "MatMul", "1", {mlp_in, w_gate}, {gate},
                   {ir::make_bool_attr("transpose_b", true)});
        b.add_node(pfx + "ffn_up", "MatMul", "1", {mlp_in, w_up}, {up},
                   {ir::make_bool_attr("transpose_b", true)});
        TensorId gate_act =
            b.add_tensor(pfx + "gate_act", DType::F32, Shape{{1, inter_dim}});
        b.add_node(pfx + "silu", (arch == "gemma" || arch == "gemma2" || arch == "gemma3") ? "Gelu"
                                                                                           : "Silu",
                   "1", {gate}, {gate_act});
        TensorId ff_hid =
            b.add_tensor(pfx + "ff_hid", DType::F32, Shape{{1, inter_dim}});
        b.add_node(pfx + "ff_mul", "Mul", "1", {gate_act, up}, {ff_hid});
        TensorId ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "ffn_down", "MatMul", "1", {ff_hid, w_down}, {ff_out},
                   {ir::make_bool_attr("transpose_b", true)});
        TensorId resid2 = b.add_tensor(pfx + "resid2", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "add2", "Add", "1", {resid1, ff_out}, {resid2});
        features = resid2;
      } else if (w_up && w_down) {
        // Dense GELU MLP (no gate) — used by some non-LLaMA GGUF exports.
        const GgufTensorInfo* up_info = find_tensor(file, pfx + "ffn_up.weight");
        const std::int64_t inter_dim = inter_dim_from_up(up_info);
        TensorId up = b.add_tensor(pfx + "up", DType::F32, Shape{{1, inter_dim}});
        b.add_node(pfx + "ffn_up", "MatMul", "1", {mlp_in, w_up}, {up},
                   {ir::make_bool_attr("transpose_b", true)});
        TensorId up_act = b.add_tensor(pfx + "up_act", DType::F32, Shape{{1, inter_dim}});
        b.add_node(pfx + "gelu", "Gelu", "1", {up}, {up_act});
        TensorId ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "ffn_down", "MatMul", "1", {up_act, w_down}, {ff_out},
                   {ir::make_bool_attr("transpose_b", true)});
        TensorId resid2 = b.add_tensor(pfx + "resid2", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "add2", "Add", "1", {resid1, ff_out}, {resid2});
        features = resid2;
      } else {
        features = resid1;
      }
    }
    if (any_mla) b.set_metadata("mla", "1");

    if (norm) {
      const GgufTensorInfo* norm_info = norm;
      if (!gguf_type_supported(norm_info->type)) {
        return Error::make(ErrorCode::NotImplemented,
                           "unsupported GGUF quant type for weight " + norm_info->name +
                               ": " + to_string(norm_info->type));
      }
      Shape nshape = dims_to_shape(norm->dims);
      TensorId nw = b.add_weight(norm->name, DType::F32, nshape, path + "#" + norm->name,
                                 gguf_type_to_quant(norm_info->type));
      const GgufTensorInfo* norm_bias = find_tensor(file, "output_norm.bias");
      TensorId nb = 0;
      if (norm_bias && gguf_type_supported(norm_bias->type)) {
        nb = b.add_weight(norm_bias->name, DType::F32, dims_to_shape(norm_bias->dims),
                          path + "#" + norm_bias->name, gguf_type_to_quant(norm_bias->type));
      }
      TensorId nout = b.add_tensor("normed", DType::F32, Shape{{1, dim}});
      if (use_layernorm && nb) {
        b.add_node("ln_out", "LayerNorm", "1", {features, nw, nb}, {nout},
                   {ir::make_float_attr("eps", layer_eps)});
      } else {
        b.add_node("rms_out", "RMSNorm", "1", {features, nw}, {nout},
                   {ir::make_float_attr("eps", rms_eps)});
      }
      features = nout;
    }

    if (!gguf_type_supported(lm->type)) {
      return Error::make(ErrorCode::NotImplemented,
                         "unsupported GGUF quant type for weight " + lm->name + ": " +
                             to_string(lm->type));
    }
    // When embeddings are tied, reuse the same weight_ref for lm_head.
    TensorId lm_w =
        tied_embeddings
            ? emb_w
            : b.add_weight(lm->name, DType::F32, lm_shape, path + "#" + lm->name,
                           gguf_type_to_quant(lm->type));
    TensorId logits = b.add_tensor("logits", DType::F32, Shape{{1, vocab}});
    b.add_node("lm_head", "MatMul", "1", {features, lm_w}, {logits},
               {ir::make_bool_attr("transpose_b", true)});
    TensorId probs = b.add_tensor("probs", DType::F32, Shape{{1, vocab}});
    b.add_node("softmax", "Softmax", "1", {logits}, {probs},
               {ir::make_int_attr("axis", -1)});
    b.set_inputs({tokens}).set_outputs({probs});
    if (arch.empty()) {
      b.set_metadata("architecture", has_blk0 ? "transformer" : "tiny_lm");
    }
    b.set_metadata("n_layers", std::to_string(n_layers));
    b.set_metadata("n_heads", std::to_string(n_heads));
    b.set_metadata("n_kv_heads", std::to_string(any_mla ? n_heads : n_kv_heads));
    b.set_metadata("head_dim", std::to_string(head_dim));
    b.set_metadata("q_dim", std::to_string(q_dim));
    if (any_mla) {
      std::int64_t mla_k_dim = 0;
      std::int64_t mla_v_dim = 0;
      if (const auto* qb = find_tensor(file, "blk.0.attn_q_b.weight")) {
        mla_k_dim = weight_out_dim(qb);
      } else if (const auto* qw = find_tensor(file, "blk.0.attn_q.weight")) {
        mla_k_dim = weight_out_dim(qw);
      }
      if (const auto* vb = find_tensor(file, "blk.0.attn_v_b.weight")) {
        mla_v_dim = weight_out_dim(vb);
      }
      if (mla_k_dim > 0) b.set_metadata("kv_dim", std::to_string(mla_k_dim));
      else b.set_metadata("kv_dim", std::to_string(kv_dim));
      if (mla_v_dim > 0) b.set_metadata("v_dim", std::to_string(mla_v_dim));
    } else {
      b.set_metadata("kv_dim", std::to_string(kv_dim));
    }
    b.set_metadata("rope_theta", std::to_string(rope_theta));
    b.set_metadata("rope_mode", std::to_string(rope_mode));
    b.set_metadata("chat_template", chat_tpl);
    b.set_metadata("norm_type", use_layernorm ? "layer" : "rms");
    if (attn_softcap > 0.f) {
      b.set_metadata("attn_logit_softcap", std::to_string(attn_softcap));
    }
    b.set_metadata("embedding_length", std::to_string(dim));
    b.set_metadata("rms_eps", std::to_string(rms_eps));
    const std::int64_t ctx = kv_i64_arch(file, arch, "context_length", 0);
    if (ctx > 0) {
      b.set_metadata("max_context", std::to_string(ctx));
      b.set_metadata("context_length", std::to_string(ctx));
    }
  } else {
    TensorId x = b.add_tensor("x", DType::F32, Shape{{1, 1}});
    TensorId y = b.add_tensor("y", DType::F32, Shape{{1, 1}});
    for (const auto& t : file.tensors) {
      Shape shape = dims_to_shape(t.dims);
      DType dt = gguf_to_dtype(t.type);
      if (dt == DType::Unknown) dt = DType::F32;
      (void)b.add_weight(t.name, dt == DType::F16 ? DType::F32 : dt, shape,
                         path + "#" + t.name);
    }
    b.add_node("id", "Identity", "1", {x}, {y});
    b.set_inputs({x}).set_outputs({y});
  }

  *out_graph = b.build();
  log::info("gguf") << "loaded " << file.tensors.size() << " tensors from " << path
                    << " graph=" << out_graph->name;
  return Error::success();
}

Error gguf_inspect_file(const std::string& path, GgufCapability* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "GgufCapability out null");
  }
  *out = GgufCapability{};
  GgufFile file;
  Error err = gguf_read_header(path, &file);
  if (!err.ok()) return err;

  auto kv_str = [&](const char* key) -> std::string {
    auto it = file.kv.find(key);
    if (it == file.kv.end()) return {};
    if (std::holds_alternative<std::string>(it->second)) return std::get<std::string>(it->second);
    return {};
  };
  out->architecture = kv_str("general.architecture");
  bool mla = false;
  bool moe = false;
  for (const auto& t : file.tensors) {
    if (t.name.find("attn_kv_a") != std::string::npos ||
        t.name.find("attn_k_b") != std::string::npos ||
        t.name.find("attn_v_b") != std::string::npos) {
      mla = true;
    }
    if (t.name.find("ffn_gate_inp") != std::string::npos ||
        t.name.find("ffn_gate_exps") != std::string::npos) {
      moe = true;
    }
  }
  out->mla = mla;
  out->moe = moe;
  const std::string a = out->architecture;
  std::string al = a;
  for (char& c : al) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (mla || al.find("deepseek") != std::string::npos || al.find("kimi") != std::string::npos) {
    out->family = "kimi_mla";
  } else if (!a.empty()) {
    out->family = al;
  } else {
    out->family = "gguf";
  }
  if (mla) {
    out->notes.push_back("MLA expand-then-attend (use GGUF: uaii generate --model kimi.gguf)");
  }
  return Error::success();
}

std::string gguf_format_capability(const GgufCapability& cap) {
  std::ostringstream os;
  os << "format: gguf\n";
  os << "family=" << cap.family << "\n";
  os << "architecture: " << (cap.architecture.empty() ? "unknown" : cap.architecture) << "\n";
  os << "mla: " << (cap.mla ? "yes" : "no") << "\n";
  os << "moe: " << (cap.moe ? "yes" : "no") << "\n";
  for (const auto& n : cap.notes) os << "note: " << n << "\n";
  if (cap.ok()) {
    os << "status: ok\n";
  } else {
    os << "status: blocked\n";
    for (const auto& b : cap.blocked) os << "blocked: " << b << "\n";
  }
  return os.str();
}

}  // namespace loaders
}  // namespace uaii
