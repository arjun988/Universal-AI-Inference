#include "uaii/loaders/gguf.hpp"

#include "binary_io.hpp"
#include "uaii/core/log.hpp"
#include "uaii/ir/graph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

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
    case GgufType::I8: return "i8";
    case GgufType::I16: return "i16";
    case GgufType::I32: return "i32";
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
    default: break;
  }
  return quant::QuantFormat::F32;
}

DType gguf_to_dtype(GgufType type) noexcept {
  switch (type) {
    case GgufType::F32: return DType::F32;
    case GgufType::F16: return DType::F16;
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
  const GgufTensorInfo* info = nullptr;
  for (const auto& t : file.tensors) {
    if (t.name == tensor_name) {
      info = &t;
      break;
    }
  }
  if (info == nullptr) {
    return Error::make(ErrorCode::NotFound, "GGUF tensor not found: " + tensor_name);
  }

  const std::size_t n = numel_of(info->dims);
  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(file.path, &r);
  if (!err.ok()) return err;
  r.seek(static_cast<std::size_t>(file.data_offset + info->offset));

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
  } else {
    return Error::make(ErrorCode::NotImplemented,
                       "unsupported GGUF dtype for f32 load: " +
                           std::string(to_string(info->type)) +
                           " (supported: F32/F16/Q4_0/Q8_0/I8)");
  }

  if (out_shape) {
    *out_shape = dims_to_shape(info->dims);
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
  const GgufTensorInfo* info = nullptr;
  for (const auto& t : file.tensors) {
    if (t.name == tensor_name) {
      info = &t;
      break;
    }
  }
  if (info == nullptr) {
    return Error::make(ErrorCode::NotFound, "tensor not in gguf: " + tensor_name);
  }
  const std::size_t n = numel_of(info->dims);
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
  detail::ByteReader r({});
  Error err = detail::ByteReader::from_file(file.path, &r);
  if (!err.ok()) return err;
  r.seek(static_cast<std::size_t>(file.data_offset + info->offset));
  out->resize(nbytes);
  err = r.read_bytes(out->data(), nbytes);
  if (!err.ok()) return err;
  if (out_shape) *out_shape = dims_to_shape(info->dims);
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
                        find_tensor(file, "blk.0.attn_norm.weight") != nullptr;
  const std::string arch = lower_ascii(kv_string(file, "general.architecture"));

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
    const std::int64_t head_dim =
        n_heads > 0 ? (emb_len > 0 ? emb_len : dim) / n_heads : dim;
    const float rope_theta = kv_float_arch(file, arch, "rope.freq_base", 10000.f);
    const float rms_eps =
        kv_float_arch(file, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
    std::int64_t n_layers = kv_i64_arch(file, arch, "block_count", 0);
    if (n_layers <= 0 && has_blk0) {
      // Infer layer count from tensor names.
      for (std::int64_t i = 0; i < 512; ++i) {
        const std::string probe = "blk." + std::to_string(i) + ".attn_norm.weight";
        if (!find_tensor(file, probe) &&
            !find_tensor(file, "blk." + std::to_string(i) + ".attn_q.weight")) {
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
        return Error::make(
            ErrorCode::NotImplemented,
            "unsupported GGUF quant type for weight " + name + ": " + to_string(ti->type));
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
      TensorId w_gate_exps = 0;
      TensorId w_up_exps = 0;
      TensorId w_down_exps = 0;
      TensorId w_gate_shexp = 0;
      TensorId w_up_shexp = 0;
      TensorId w_down_shexp = 0;
      err = add_weight_ref(pfx + "attn_norm.weight", &attn_norm);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_q.weight", &wq);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_k.weight", &wk);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_v.weight", &wv);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "attn_output.weight", &wo);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_norm.weight", &ffn_norm);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_gate.weight", &w_gate);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_up.weight", &w_up);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_down.weight", &w_down);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_gate_inp.weight", &w_gate_inp);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_gate_exps.weight", &w_gate_exps);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_up_exps.weight", &w_up_exps);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_down_exps.weight", &w_down_exps);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_gate_shexp.weight", &w_gate_shexp);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_up_shexp.weight", &w_up_shexp);
      if (!err.ok()) return err;
      err = add_weight_ref(pfx + "ffn_down_shexp.weight", &w_down_shexp);
      if (!err.ok()) return err;

      if (!attn_norm || !wq || !wk || !wv || !wo) {
        // Incomplete layer — stop stacking further blocks.
        break;
      }

      TensorId n1 = b.add_tensor(pfx + "n1", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "rms1", "RMSNorm", "1", {features, attn_norm}, {n1},
                 {ir::make_float_attr("eps", rms_eps)});

      TensorId q = b.add_tensor(pfx + "q", DType::F32, Shape{{1, dim}});
      TensorId k = b.add_tensor(pfx + "k", DType::F32, Shape{{1, dim}});
      TensorId v = b.add_tensor(pfx + "v", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "q_proj", "MatMul", "1", {n1, wq}, {q},
                 {ir::make_bool_attr("transpose_b", true)});
      b.add_node(pfx + "k_proj", "MatMul", "1", {n1, wk}, {k},
                 {ir::make_bool_attr("transpose_b", true)});
      b.add_node(pfx + "v_proj", "MatMul", "1", {n1, wv}, {v},
                 {ir::make_bool_attr("transpose_b", true)});

      TensorId q_rope = b.add_tensor(pfx + "q_rope", DType::F32, Shape{{1, dim}});
      TensorId k_rope = b.add_tensor(pfx + "k_rope", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "q_rope", "RoPE", "1", {q}, {q_rope},
                 {ir::make_float_attr("theta", rope_theta)});
      b.add_node(pfx + "k_rope", "RoPE", "1", {k}, {k_rope},
                 {ir::make_float_attr("theta", rope_theta)});

      TensorId attn = b.add_tensor(pfx + "attn", DType::F32, Shape{{1, dim}});
      // kv_heads is recorded for GQA metadata; Attention kernel uses num_heads today.
      b.add_node(pfx + "attn", "Attention", "1", {q_rope, k_rope, v}, {attn},
                 {ir::make_int_attr("num_heads", n_heads),
                  ir::make_int_attr("kv_heads", n_kv_heads),
                  ir::make_bool_attr("causal", true),
                  ir::make_bool_attr("use_kv_cache", true),
                  ir::make_int_attr("layer_id", li)});

      TensorId attn_o = b.add_tensor(pfx + "attn_o", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "o_proj", "MatMul", "1", {attn, wo}, {attn_o},
                 {ir::make_bool_attr("transpose_b", true)});

      TensorId resid1 = b.add_tensor(pfx + "resid1", DType::F32, Shape{{1, dim}});
      b.add_node(pfx + "add1", "Add", "1", {features, attn_o}, {resid1});

      TensorId mlp_in = resid1;
      if (ffn_norm) {
        TensorId n2 = b.add_tensor(pfx + "n2", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "rms2", "RMSNorm", "1", {resid1, ffn_norm}, {n2},
                   {ir::make_float_attr("eps", rms_eps)});
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

      if (w_gate_inp && w_gate_exps && w_up_exps && w_down_exps) {
        // Mixtral / Qwen2MoE-style: router + top-k SwiGLU experts (+ optional shared).
        const GgufTensorInfo* exps_info = find_tensor(file, pfx + "ffn_gate_exps.weight");
        Shape exps_shape = exps_info ? dims_to_shape(exps_info->dims) : Shape{};
        std::int64_t n_experts = kv_i64_arch(file, arch, "expert_count", 0);
        if (n_experts <= 0 && exps_shape.dims.size() >= 1) n_experts = exps_shape.dims[0];
        if (n_experts <= 0) {
          return Error::make(ErrorCode::InvalidArgument,
                             "MoE layer " + pfx + " missing expert_count / expert dims");
        }
        std::int64_t top_k = kv_i64_arch(file, arch, "expert_used_count", 2);
        if (top_k <= 0) top_k = 2;
        if (top_k > n_experts) top_k = n_experts;

        TensorId router_probs =
            b.add_tensor(pfx + "moe_probs", DType::F32, Shape{{1, n_experts}});
        TensorId router_top =
            b.add_tensor(pfx + "moe_top", DType::F32, Shape{{1, 1}});
        b.add_node(pfx + "moe_router", "MoERouter", "1", {mlp_in, w_gate_inp},
                   {router_probs, router_top},
                   {ir::make_int_attr("num_experts", n_experts),
                    ir::make_int_attr("top_k", top_k)});

        TensorId ff_out = b.add_tensor(pfx + "ff_out", DType::F32, Shape{{1, dim}});
        b.add_node(pfx + "moe_exps", "MoEExpertsSwiGLU", "1",
                   {mlp_in, w_gate_exps, w_up_exps, w_down_exps, router_probs}, {ff_out},
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
        b.add_node(pfx + "silu", "Silu", "1", {gate}, {gate_act});
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
      TensorId nout = b.add_tensor("normed", DType::F32, Shape{{1, dim}});
      b.add_node("rms_out", "RMSNorm", "1", {features, nw}, {nout},
                 {ir::make_float_attr("eps", rms_eps)});
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
    b.set_metadata("n_kv_heads", std::to_string(n_kv_heads));
    b.set_metadata("head_dim", std::to_string(head_dim));
    b.set_metadata("rope_theta", std::to_string(rope_theta));
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

}  // namespace loaders
}  // namespace uaii
