#include "uaii/ir/serialize.hpp"

#include "json_util.hpp"
#include "uaii/ir/dtype.hpp"

#include <sstream>

namespace uaii {
namespace ir {
namespace {

json::Value shape_to_json(const Shape& shape) {
  json::Array arr;
  for (std::int64_t d : shape.dims) {
    arr.push_back(json::Value::num(static_cast<double>(d)));
  }
  return json::Value::array(std::move(arr));
}

Error shape_from_json(const json::Value& v, Shape* out) {
  const json::Array* arr = nullptr;
  Error err = json::require_array(v, &arr, "shape");
  if (!err.ok()) {
    return err;
  }
  out->dims.clear();
  for (const auto& item : *arr) {
    if (!item.is_number()) {
      return Error::make(ErrorCode::InvalidArgument, "shape dims must be numbers");
    }
    out->dims.push_back(static_cast<std::int64_t>(item.number));
  }
  return Error::ok();
}

json::Value attr_to_json(const Attribute& attr) {
  json::Object o;
  o["key"] = json::Value::string(attr.key);
  o["type"] = json::Value::string(to_string(attr.type));
  switch (attr.type) {
    case AttributeType::String:
      o["value"] = json::Value::string(std::get<std::string>(attr.value));
      break;
    case AttributeType::Int:
      o["value"] = json::Value::num(static_cast<double>(std::get<std::int64_t>(attr.value)));
      break;
    case AttributeType::Float:
      o["value"] = json::Value::num(std::get<double>(attr.value));
      break;
    case AttributeType::Bool:
      o["value"] = json::Value::boolean(std::get<bool>(attr.value));
      break;
    case AttributeType::IntArray: {
      json::Array a;
      for (std::int64_t x : std::get<std::vector<std::int64_t>>(attr.value)) {
        a.push_back(json::Value::num(static_cast<double>(x)));
      }
      o["value"] = json::Value::array(std::move(a));
      break;
    }
    case AttributeType::FloatArray: {
      json::Array a;
      for (double x : std::get<std::vector<double>>(attr.value)) {
        a.push_back(json::Value::num(x));
      }
      o["value"] = json::Value::array(std::move(a));
      break;
    }
  }
  return json::Value::object(std::move(o));
}

Error attr_from_json(const json::Value& v, Attribute* out) {
  const json::Object* o = nullptr;
  Error err = json::require_object(v, &o, "attribute");
  if (!err.ok()) {
    return err;
  }
  const json::Value* key = json::get(*o, "key");
  const json::Value* type = json::get(*o, "type");
  const json::Value* value = json::get(*o, "value");
  if (key == nullptr || !key->is_string() || type == nullptr || !type->is_string() ||
      value == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "attribute missing key/type/value");
  }
  out->key = key->str;
  const std::string& t = type->str;
  if (t == "string") {
    if (!value->is_string()) {
      return Error::make(ErrorCode::InvalidArgument, "string attr value");
    }
    *out = make_string_attr(out->key, value->str);
  } else if (t == "int") {
    if (!value->is_number()) {
      return Error::make(ErrorCode::InvalidArgument, "int attr value");
    }
    *out = make_int_attr(out->key, static_cast<std::int64_t>(value->number));
  } else if (t == "float") {
    if (!value->is_number()) {
      return Error::make(ErrorCode::InvalidArgument, "float attr value");
    }
    *out = make_float_attr(out->key, value->number);
  } else if (t == "bool") {
    if (!value->is_bool()) {
      return Error::make(ErrorCode::InvalidArgument, "bool attr value");
    }
    *out = make_bool_attr(out->key, value->b);
  } else if (t == "int_array") {
    const json::Array* arr = nullptr;
    err = json::require_array(*value, &arr, "int_array");
    if (!err.ok()) {
      return err;
    }
    Attribute a;
    a.key = out->key;
    a.type = AttributeType::IntArray;
    std::vector<std::int64_t> vals;
    for (const auto& item : *arr) {
      if (!item.is_number()) {
        return Error::make(ErrorCode::InvalidArgument, "int_array item");
      }
      vals.push_back(static_cast<std::int64_t>(item.number));
    }
    a.value = std::move(vals);
    *out = std::move(a);
  } else if (t == "float_array") {
    const json::Array* arr = nullptr;
    err = json::require_array(*value, &arr, "float_array");
    if (!err.ok()) {
      return err;
    }
    Attribute a;
    a.key = out->key;
    a.type = AttributeType::FloatArray;
    std::vector<double> vals;
    for (const auto& item : *arr) {
      if (!item.is_number()) {
        return Error::make(ErrorCode::InvalidArgument, "float_array item");
      }
      vals.push_back(item.number);
    }
    a.value = std::move(vals);
    *out = std::move(a);
  } else {
    return Error::make(ErrorCode::InvalidArgument, "unknown attribute type " + t);
  }
  return Error::ok();
}

}  // namespace

Error graph_to_json(const Graph& graph, std::string* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "json out is null");
  }

  json::Object root;
  json::Object ver;
  ver["major"] = json::Value::num(graph.version.major);
  ver["minor"] = json::Value::num(graph.version.minor);
  root["ir_version"] = json::Value::object(std::move(ver));
  root["name"] = json::Value::string(graph.name);
  root["producer"] = json::Value::string(graph.producer);
  root["domain"] = json::Value::string(graph.domain);

  json::Array tensors;
  for (const auto& t : graph.tensors) {
    json::Object o;
    o["id"] = json::Value::num(static_cast<double>(t.id));
    o["name"] = json::Value::string(t.name);
    o["dtype"] = json::Value::string(uaii::to_string(t.dtype));
    o["shape"] = shape_to_json(t.shape);
    o["storage_hint"] = json::Value::string(to_string(t.storage_hint));
    o["is_weight"] = json::Value::boolean(t.is_weight);
    o["weight_ref"] = json::Value::string(t.weight_ref);
    tensors.push_back(json::Value::object(std::move(o)));
  }
  root["tensors"] = json::Value::array(std::move(tensors));

  json::Array nodes;
  for (const auto& n : graph.nodes) {
    json::Object o;
    o["id"] = json::Value::num(static_cast<double>(n.id));
    o["name"] = json::Value::string(n.name);
    o["op_name"] = json::Value::string(n.op_name);
    o["op_version"] = json::Value::string(n.op_version);
    json::Array ins;
    for (TensorId id : n.inputs) {
      ins.push_back(json::Value::num(static_cast<double>(id)));
    }
    json::Array outs;
    for (TensorId id : n.outputs) {
      outs.push_back(json::Value::num(static_cast<double>(id)));
    }
    o["inputs"] = json::Value::array(std::move(ins));
    o["outputs"] = json::Value::array(std::move(outs));
    json::Array attrs;
    for (const auto& a : n.attributes) {
      attrs.push_back(attr_to_json(a));
    }
    o["attributes"] = json::Value::array(std::move(attrs));
    nodes.push_back(json::Value::object(std::move(o)));
  }
  root["nodes"] = json::Value::array(std::move(nodes));

  json::Array gins;
  for (TensorId id : graph.inputs) {
    gins.push_back(json::Value::num(static_cast<double>(id)));
  }
  json::Array gouts;
  for (TensorId id : graph.outputs) {
    gouts.push_back(json::Value::num(static_cast<double>(id)));
  }
  root["inputs"] = json::Value::array(std::move(gins));
  root["outputs"] = json::Value::array(std::move(gouts));

  json::Object meta;
  for (const auto& kv : graph.metadata) {
    meta[kv.first] = json::Value::string(kv.second);
  }
  root["metadata"] = json::Value::object(std::move(meta));

  *out = json::stringify(json::Value::object(std::move(root)), true);
  return Error::ok();
}

Error graph_from_json(const std::string& text, Graph* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "graph out is null");
  }
  json::Value root_v;
  Error err = json::parse(text, &root_v);
  if (!err.ok()) {
    return err;
  }
  const json::Object* root = nullptr;
  err = json::require_object(root_v, &root, "graph");
  if (!err.ok()) {
    return err;
  }

  Graph g;
  if (const json::Value* ver = json::get(*root, "ir_version")) {
    const json::Object* vo = nullptr;
    err = json::require_object(*ver, &vo, "ir_version");
    if (!err.ok()) {
      return err;
    }
    const json::Value* maj = json::get(*vo, "major");
    const json::Value* min = json::get(*vo, "minor");
    if (maj == nullptr || !maj->is_number() || min == nullptr || !min->is_number()) {
      return Error::make(ErrorCode::InvalidArgument, "ir_version.major/minor required");
    }
    g.version.major = static_cast<std::uint32_t>(maj->number);
    g.version.minor = static_cast<std::uint32_t>(min->number);
  }

  if (const json::Value* v = json::get(*root, "name"); v && v->is_string()) {
    g.name = v->str;
  }
  if (const json::Value* v = json::get(*root, "producer"); v && v->is_string()) {
    g.producer = v->str;
  }
  if (const json::Value* v = json::get(*root, "domain"); v && v->is_string()) {
    g.domain = v->str;
  }

  if (const json::Value* tv = json::get(*root, "tensors")) {
    const json::Array* arr = nullptr;
    err = json::require_array(*tv, &arr, "tensors");
    if (!err.ok()) {
      return err;
    }
    for (const auto& item : *arr) {
      const json::Object* o = nullptr;
      err = json::require_object(item, &o, "tensor");
      if (!err.ok()) {
        return err;
      }
      Tensor t;
      const json::Value* id = json::get(*o, "id");
      if (id == nullptr || !id->is_number()) {
        return Error::make(ErrorCode::InvalidArgument, "tensor.id required");
      }
      t.id = static_cast<TensorId>(id->number);
      if (const json::Value* n = json::get(*o, "name"); n && n->is_string()) {
        t.name = n->str;
      }
      if (const json::Value* d = json::get(*o, "dtype"); d && d->is_string()) {
        if (!parse_dtype(d->str, &t.dtype)) {
          return Error::make(ErrorCode::InvalidArgument, "bad dtype " + d->str);
        }
      }
      if (const json::Value* s = json::get(*o, "shape")) {
        err = shape_from_json(*s, &t.shape);
        if (!err.ok()) {
          return err;
        }
      }
      if (const json::Value* h = json::get(*o, "storage_hint"); h && h->is_string()) {
        if (!parse_storage_hint(h->str, &t.storage_hint)) {
          return Error::make(ErrorCode::InvalidArgument, "bad storage_hint");
        }
      }
      if (const json::Value* w = json::get(*o, "is_weight"); w && w->is_bool()) {
        t.is_weight = w->b;
      }
      if (const json::Value* r = json::get(*o, "weight_ref"); r && r->is_string()) {
        t.weight_ref = r->str;
      }
      g.tensors.push_back(std::move(t));
    }
  }

  if (const json::Value* nv = json::get(*root, "nodes")) {
    const json::Array* arr = nullptr;
    err = json::require_array(*nv, &arr, "nodes");
    if (!err.ok()) {
      return err;
    }
    for (const auto& item : *arr) {
      const json::Object* o = nullptr;
      err = json::require_object(item, &o, "node");
      if (!err.ok()) {
        return err;
      }
      Node n;
      const json::Value* id = json::get(*o, "id");
      if (id == nullptr || !id->is_number()) {
        return Error::make(ErrorCode::InvalidArgument, "node.id required");
      }
      n.id = static_cast<NodeId>(id->number);
      if (const json::Value* name = json::get(*o, "name"); name && name->is_string()) {
        n.name = name->str;
      }
      if (const json::Value* op = json::get(*o, "op_name"); op && op->is_string()) {
        n.op_name = op->str;
      }
      if (const json::Value* ov = json::get(*o, "op_version"); ov && ov->is_string()) {
        n.op_version = ov->str;
      } else {
        n.op_version = "1";
      }
      if (const json::Value* ins = json::get(*o, "inputs")) {
        const json::Array* a = nullptr;
        err = json::require_array(*ins, &a, "node.inputs");
        if (!err.ok()) {
          return err;
        }
        for (const auto& x : *a) {
          if (!x.is_number()) {
            return Error::make(ErrorCode::InvalidArgument, "node.inputs id");
          }
          n.inputs.push_back(static_cast<TensorId>(x.number));
        }
      }
      if (const json::Value* outs = json::get(*o, "outputs")) {
        const json::Array* a = nullptr;
        err = json::require_array(*outs, &a, "node.outputs");
        if (!err.ok()) {
          return err;
        }
        for (const auto& x : *a) {
          if (!x.is_number()) {
            return Error::make(ErrorCode::InvalidArgument, "node.outputs id");
          }
          n.outputs.push_back(static_cast<TensorId>(x.number));
        }
      }
      if (const json::Value* attrs = json::get(*o, "attributes")) {
        const json::Array* a = nullptr;
        err = json::require_array(*attrs, &a, "node.attributes");
        if (!err.ok()) {
          return err;
        }
        for (const auto& av : *a) {
          Attribute attr;
          err = attr_from_json(av, &attr);
          if (!err.ok()) {
            return err;
          }
          n.attributes.push_back(std::move(attr));
        }
      }
      g.nodes.push_back(std::move(n));
    }
  }

  auto read_id_array = [&](const char* key, std::vector<TensorId>* dest) -> Error {
    const json::Value* v = json::get(*root, key);
    if (v == nullptr) {
      return Error::ok();
    }
    const json::Array* a = nullptr;
    Error e = json::require_array(*v, &a, key);
    if (!e.ok()) {
      return e;
    }
    for (const auto& x : *a) {
      if (!x.is_number()) {
        return Error::make(ErrorCode::InvalidArgument, std::string(key) + " ids");
      }
      dest->push_back(static_cast<TensorId>(x.number));
    }
    return Error::ok();
  };

  err = read_id_array("inputs", &g.inputs);
  if (!err.ok()) {
    return err;
  }
  err = read_id_array("outputs", &g.outputs);
  if (!err.ok()) {
    return err;
  }

  if (const json::Value* meta = json::get(*root, "metadata"); meta && meta->is_object()) {
    for (const auto& kv : meta->obj) {
      if (kv.second.is_string()) {
        g.metadata[kv.first] = kv.second.str;
      }
    }
  }

  *out = std::move(g);
  return Error::ok();
}

std::string plan_to_json(const ExecutionPlan& plan) {
  json::Object root;
  json::Object ver;
  ver["major"] = json::Value::num(plan.ir_version.major);
  ver["minor"] = json::Value::num(plan.ir_version.minor);
  root["ir_version"] = json::Value::object(std::move(ver));
  root["graph_name"] = json::Value::string(plan.graph_name);

  json::Array ops;
  for (const auto& op : plan.ops) {
    json::Object o;
    o["node_id"] = json::Value::num(static_cast<double>(op.node_id));
    o["node_name"] = json::Value::string(op.node_name);
    o["op_name"] = json::Value::string(op.op_name);
    o["op_version"] = json::Value::string(op.op_version);
    o["preferred_device"] = json::Value::string(uaii::to_string(op.preferred_device));
    o["selected_kernel"] = json::Value::string(op.selected_kernel);
    json::Array deps;
    for (NodeId d : op.dependencies) {
      deps.push_back(json::Value::num(static_cast<double>(d)));
    }
    o["dependencies"] = json::Value::array(std::move(deps));
    json::Array ins;
    for (TensorId id : op.inputs) {
      ins.push_back(json::Value::num(static_cast<double>(id)));
    }
    json::Array outs;
    for (TensorId id : op.outputs) {
      outs.push_back(json::Value::num(static_cast<double>(id)));
    }
    o["inputs"] = json::Value::array(std::move(ins));
    o["outputs"] = json::Value::array(std::move(outs));
    ops.push_back(json::Value::object(std::move(o)));
  }
  root["ops"] = json::Value::array(std::move(ops));
  return json::stringify(json::Value::object(std::move(root)), true);
}

}  // namespace ir
}  // namespace uaii
