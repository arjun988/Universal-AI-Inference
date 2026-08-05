#include "uaii/ir/serialize.hpp"

#include "uaii/ir/dtype.hpp"

#include <cstring>

namespace uaii {
namespace ir {
namespace {

// Native binary layout aligned with schemas/uaii_ir.fbs logical model.
// Header: magic[4] + format_version_u32(=1) + ir_major_u32 + ir_minor_u32
// Then length-prefixed fields (little-endian).

class Writer {
 public:
  void u8(std::uint8_t v) { data_.push_back(static_cast<char>(v)); }
  void u32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
      data_.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
    }
  }
  void u64(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      data_.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
    }
  }
  void i64(std::int64_t v) { u64(static_cast<std::uint64_t>(v)); }
  void f64(double v) {
    std::uint64_t bits = 0;
    static_assert(sizeof(double) == sizeof(std::uint64_t), "unexpected double size");
    std::memcpy(&bits, &v, sizeof(bits));
    u64(bits);
  }
  void str(const std::string& s) {
    u32(static_cast<std::uint32_t>(s.size()));
    data_.insert(data_.end(), s.begin(), s.end());
  }
  void bytes(const std::string& s) { data_.append(s); }
  [[nodiscard]] std::string& data() { return data_; }

 private:
  std::string data_;
};

class Reader {
 public:
  explicit Reader(const std::string& data) : data_(data) {}

  Error u8(std::uint8_t* v) {
    if (pos_ + 1 > data_.size()) {
      return eof();
    }
    *v = static_cast<std::uint8_t>(data_[pos_++]);
    return Error::success();
  }
  Error u32(std::uint32_t* v) {
    if (pos_ + 4 > data_.size()) {
      return eof();
    }
    std::uint32_t x = 0;
    for (int i = 0; i < 4; ++i) {
      x |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(data_[pos_++])) << (8 * i);
    }
    *v = x;
    return Error::success();
  }
  Error u64(std::uint64_t* v) {
    if (pos_ + 8 > data_.size()) {
      return eof();
    }
    std::uint64_t x = 0;
    for (int i = 0; i < 8; ++i) {
      x |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(data_[pos_++])) << (8 * i);
    }
    *v = x;
    return Error::success();
  }
  Error i64(std::int64_t* v) {
    std::uint64_t x = 0;
    Error err = u64(&x);
    if (!err.ok()) {
      return err;
    }
    *v = static_cast<std::int64_t>(x);
    return Error::success();
  }
  Error f64(double* v) {
    std::uint64_t bits = 0;
    Error err = u64(&bits);
    if (!err.ok()) {
      return err;
    }
    std::memcpy(v, &bits, sizeof(bits));
    return Error::success();
  }
  Error str(std::string* out) {
    std::uint32_t n = 0;
    Error err = u32(&n);
    if (!err.ok()) {
      return err;
    }
    if (pos_ + n > data_.size()) {
      return eof();
    }
    out->assign(data_.data() + pos_, n);
    pos_ += n;
    return Error::success();
  }

 private:
  const std::string& data_;
  std::size_t pos_ = 0;
  Error eof() const {
    return Error::make(ErrorCode::InvalidArgument, "truncated UAII binary IR");
  }
};

void write_attr(Writer* w, const Attribute& a) {
  w->str(a.key);
  w->u8(static_cast<std::uint8_t>(a.type));
  switch (a.type) {
    case AttributeType::String:
      w->str(std::get<std::string>(a.value));
      break;
    case AttributeType::Int:
      w->i64(std::get<std::int64_t>(a.value));
      break;
    case AttributeType::Float:
      w->f64(std::get<double>(a.value));
      break;
    case AttributeType::Bool:
      w->u8(std::get<bool>(a.value) ? 1 : 0);
      break;
    case AttributeType::IntArray: {
      const auto& vals = std::get<std::vector<std::int64_t>>(a.value);
      w->u32(static_cast<std::uint32_t>(vals.size()));
      for (std::int64_t x : vals) {
        w->i64(x);
      }
      break;
    }
    case AttributeType::FloatArray: {
      const auto& vals = std::get<std::vector<double>>(a.value);
      w->u32(static_cast<std::uint32_t>(vals.size()));
      for (double x : vals) {
        w->f64(x);
      }
      break;
    }
  }
}

Error read_attr(Reader* r, Attribute* a) {
  Error err = r->str(&a->key);
  if (!err.ok()) {
    return err;
  }
  std::uint8_t type = 0;
  err = r->u8(&type);
  if (!err.ok()) {
    return err;
  }
  a->type = static_cast<AttributeType>(type);
  switch (a->type) {
    case AttributeType::String: {
      std::string s;
      err = r->str(&s);
      if (!err.ok()) return err;
      a->value = std::move(s);
      break;
    }
    case AttributeType::Int: {
      std::int64_t v = 0;
      err = r->i64(&v);
      if (!err.ok()) return err;
      a->value = v;
      break;
    }
    case AttributeType::Float: {
      double v = 0;
      err = r->f64(&v);
      if (!err.ok()) return err;
      a->value = v;
      break;
    }
    case AttributeType::Bool: {
      std::uint8_t v = 0;
      err = r->u8(&v);
      if (!err.ok()) return err;
      a->value = (v != 0);
      break;
    }
    case AttributeType::IntArray: {
      std::uint32_t n = 0;
      err = r->u32(&n);
      if (!err.ok()) return err;
      std::vector<std::int64_t> vals(n);
      for (std::uint32_t i = 0; i < n; ++i) {
        err = r->i64(&vals[i]);
        if (!err.ok()) return err;
      }
      a->value = std::move(vals);
      break;
    }
    case AttributeType::FloatArray: {
      std::uint32_t n = 0;
      err = r->u32(&n);
      if (!err.ok()) return err;
      std::vector<double> vals(n);
      for (std::uint32_t i = 0; i < n; ++i) {
        err = r->f64(&vals[i]);
        if (!err.ok()) return err;
      }
      a->value = std::move(vals);
      break;
    }
    default:
      return Error::make(ErrorCode::InvalidArgument, "unknown attribute type in binary IR");
  }
  return Error::success();
}

}  // namespace

Error graph_to_binary(const Graph& graph, std::string* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "binary out is null");
  }
  Writer w;
  w.bytes(std::string(kBinaryMagic, 4));
  w.u32(1);  // codec format version
  w.u32(graph.version.major);
  w.u32(graph.version.minor);
  w.str(graph.name);
  w.str(graph.producer);
  w.str(graph.domain);

  w.u32(static_cast<std::uint32_t>(graph.tensors.size()));
  for (const auto& t : graph.tensors) {
    w.u64(t.id);
    w.str(t.name);
    w.u8(static_cast<std::uint8_t>(t.dtype));
    w.u32(static_cast<std::uint32_t>(t.shape.dims.size()));
    for (std::int64_t d : t.shape.dims) {
      w.i64(d);
    }
    w.u8(static_cast<std::uint8_t>(t.storage_hint));
    w.u8(t.is_weight ? 1 : 0);
    w.str(t.weight_ref);
  }

  w.u32(static_cast<std::uint32_t>(graph.nodes.size()));
  for (const auto& n : graph.nodes) {
    w.u64(n.id);
    w.str(n.name);
    w.str(n.op_name);
    w.str(n.op_version);
    w.u32(static_cast<std::uint32_t>(n.inputs.size()));
    for (TensorId id : n.inputs) {
      w.u64(id);
    }
    w.u32(static_cast<std::uint32_t>(n.outputs.size()));
    for (TensorId id : n.outputs) {
      w.u64(id);
    }
    w.u32(static_cast<std::uint32_t>(n.attributes.size()));
    for (const auto& a : n.attributes) {
      write_attr(&w, a);
    }
  }

  w.u32(static_cast<std::uint32_t>(graph.inputs.size()));
  for (TensorId id : graph.inputs) {
    w.u64(id);
  }
  w.u32(static_cast<std::uint32_t>(graph.outputs.size()));
  for (TensorId id : graph.outputs) {
    w.u64(id);
  }

  w.u32(static_cast<std::uint32_t>(graph.metadata.size()));
  for (const auto& kv : graph.metadata) {
    w.str(kv.first);
    w.str(kv.second);
  }

  *out = std::move(w.data());
  return Error::success();
}

Error graph_from_binary(const std::string& bytes, Graph* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out is null");
  }
  if (bytes.size() < 16 ||
      bytes[0] != kBinaryMagic[0] || bytes[1] != kBinaryMagic[1] ||
      bytes[2] != kBinaryMagic[2] || bytes[3] != kBinaryMagic[3]) {
    return Error::make(ErrorCode::InvalidArgument, "not a UAII binary IR file");
  }

  Reader r(bytes);
  // Consume magic (already validated above).
  for (int i = 0; i < 4; ++i) {
    std::uint8_t b = 0;
    Error err = r.u8(&b);
    if (!err.ok()) {
      return err;
    }
    (void)b;
  }
  Error err;

  std::uint32_t codec = 0;
  err = r.u32(&codec);
  if (!err.ok()) return err;
  if (codec != 1) {
    return Error::make(ErrorCode::InvalidArgument,
                       "unsupported UAII binary codec version " + std::to_string(codec));
  }

  Graph g;
  err = r.u32(&g.version.major);
  if (!err.ok()) return err;
  err = r.u32(&g.version.minor);
  if (!err.ok()) return err;
  err = check_compatible(g.version);
  if (!err.ok()) return err;

  err = r.str(&g.name);
  if (!err.ok()) return err;
  err = r.str(&g.producer);
  if (!err.ok()) return err;
  err = r.str(&g.domain);
  if (!err.ok()) return err;

  std::uint32_t n_tensors = 0;
  err = r.u32(&n_tensors);
  if (!err.ok()) return err;
  g.tensors.reserve(n_tensors);
  for (std::uint32_t i = 0; i < n_tensors; ++i) {
    Tensor t;
    err = r.u64(&t.id);
    if (!err.ok()) return err;
    err = r.str(&t.name);
    if (!err.ok()) return err;
    std::uint8_t dtype = 0;
    err = r.u8(&dtype);
    if (!err.ok()) return err;
    t.dtype = static_cast<DType>(dtype);
    std::uint32_t nd = 0;
    err = r.u32(&nd);
    if (!err.ok()) return err;
    t.shape.dims.resize(nd);
    for (std::uint32_t d = 0; d < nd; ++d) {
      err = r.i64(&t.shape.dims[d]);
      if (!err.ok()) return err;
    }
    std::uint8_t hint = 0;
    err = r.u8(&hint);
    if (!err.ok()) return err;
    t.storage_hint = static_cast<StorageHint>(hint);
    std::uint8_t is_w = 0;
    err = r.u8(&is_w);
    if (!err.ok()) return err;
    t.is_weight = is_w != 0;
    err = r.str(&t.weight_ref);
    if (!err.ok()) return err;
    g.tensors.push_back(std::move(t));
  }

  std::uint32_t n_nodes = 0;
  err = r.u32(&n_nodes);
  if (!err.ok()) return err;
  g.nodes.reserve(n_nodes);
  for (std::uint32_t i = 0; i < n_nodes; ++i) {
    Node n;
    err = r.u64(&n.id);
    if (!err.ok()) return err;
    err = r.str(&n.name);
    if (!err.ok()) return err;
    err = r.str(&n.op_name);
    if (!err.ok()) return err;
    err = r.str(&n.op_version);
    if (!err.ok()) return err;
    std::uint32_t ni = 0;
    err = r.u32(&ni);
    if (!err.ok()) return err;
    n.inputs.resize(ni);
    for (std::uint32_t k = 0; k < ni; ++k) {
      err = r.u64(&n.inputs[k]);
      if (!err.ok()) return err;
    }
    std::uint32_t no = 0;
    err = r.u32(&no);
    if (!err.ok()) return err;
    n.outputs.resize(no);
    for (std::uint32_t k = 0; k < no; ++k) {
      err = r.u64(&n.outputs[k]);
      if (!err.ok()) return err;
    }
    std::uint32_t na = 0;
    err = r.u32(&na);
    if (!err.ok()) return err;
    n.attributes.resize(na);
    for (std::uint32_t k = 0; k < na; ++k) {
      err = read_attr(&r, &n.attributes[k]);
      if (!err.ok()) return err;
    }
    g.nodes.push_back(std::move(n));
  }

  std::uint32_t n_in = 0;
  err = r.u32(&n_in);
  if (!err.ok()) return err;
  g.inputs.resize(n_in);
  for (std::uint32_t i = 0; i < n_in; ++i) {
    err = r.u64(&g.inputs[i]);
    if (!err.ok()) return err;
  }
  std::uint32_t n_out = 0;
  err = r.u32(&n_out);
  if (!err.ok()) return err;
  g.outputs.resize(n_out);
  for (std::uint32_t i = 0; i < n_out; ++i) {
    err = r.u64(&g.outputs[i]);
    if (!err.ok()) return err;
  }

  std::uint32_t n_meta = 0;
  err = r.u32(&n_meta);
  if (!err.ok()) return err;
  for (std::uint32_t i = 0; i < n_meta; ++i) {
    std::string k, v;
    err = r.str(&k);
    if (!err.ok()) return err;
    err = r.str(&v);
    if (!err.ok()) return err;
    g.metadata[std::move(k)] = std::move(v);
  }

  *out = std::move(g);
  return Error::success();
}

}  // namespace ir
}  // namespace uaii
