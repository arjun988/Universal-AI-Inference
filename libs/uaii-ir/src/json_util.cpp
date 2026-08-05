#include "json_util.hpp"

#include <cctype>
#include <cmath>
#include <sstream>

namespace uaii {
namespace ir {
namespace json {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Error parse(Value* out) {
    skip_ws();
    Error err = parse_value(out);
    if (!err.ok()) {
      return err;
    }
    skip_ws();
    if (pos_ != text_.size()) {
      return fail("trailing characters after JSON value");
    }
    return Error::success();
  }

 private:
  const std::string& text_;
  std::size_t pos_ = 0;

  Error fail(const std::string& msg) const {
    return Error::make(ErrorCode::InvalidArgument,
                       "json@" + std::to_string(pos_) + ": " + msg);
  }

  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool consume(char c) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  Error parse_value(Value* out) {
    skip_ws();
    if (pos_ >= text_.size()) {
      return fail("unexpected end of input");
    }
    const char c = text_[pos_];
    if (c == 'n') return parse_null(out);
    if (c == 't' || c == 'f') return parse_bool(out);
    if (c == '"') return parse_string(out);
    if (c == '[') return parse_array(out);
    if (c == '{') return parse_object(out);
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      return parse_number(out);
    }
    return fail(std::string("unexpected character '") + c + "'");
  }

  Error parse_null(Value* out) {
    if (text_.compare(pos_, 4, "null") != 0) {
      return fail("expected null");
    }
    pos_ += 4;
    *out = Value::null();
    return Error::success();
  }

  Error parse_bool(Value* out) {
    if (text_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      *out = Value::boolean(true);
      return Error::success();
    }
    if (text_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      *out = Value::boolean(false);
      return Error::success();
    }
    return fail("expected boolean");
  }

  Error parse_number(Value* out) {
    const std::size_t start = pos_;
    if (text_[pos_] == '-') {
      ++pos_;
    }
    if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      return fail("invalid number");
    }
    if (text_[pos_] == '0') {
      ++pos_;
    } else {
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        return fail("invalid fraction");
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        return fail("invalid exponent");
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    try {
      *out = Value::num(std::stod(text_.substr(start, pos_ - start)));
    } catch (...) {
      return fail("number parse failed");
    }
    return Error::success();
  }

  Error parse_string(Value* out) {
    if (!consume('"')) {
      return fail("expected string");
    }
    std::string s;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') {
        *out = Value::string(std::move(s));
        return Error::success();
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          return fail("unterminated escape");
        }
        const char e = text_[pos_++];
        switch (e) {
          case '"':
          case '\\':
          case '/':
            s.push_back(e);
            break;
          case 'b': s.push_back('\b'); break;
          case 'f': s.push_back('\f'); break;
          case 'n': s.push_back('\n'); break;
          case 'r': s.push_back('\r'); break;
          case 't': s.push_back('\t'); break;
          case 'u': {
            if (pos_ + 4 > text_.size()) {
              return fail("invalid unicode escape");
            }
            // Minimal: keep as UTF-8 for BMP via wchar-less hex decode of codepoint < 0x80
            // For simplicity store '?' for non-ASCII escapes in Phase 2 hand-authored files.
            unsigned code = 0;
            for (int i = 0; i < 4; ++i) {
              const char h = text_[pos_++];
              code <<= 4;
              if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
              else return fail("invalid unicode escape");
            }
            if (code < 0x80) {
              s.push_back(static_cast<char>(code));
            } else if (code < 0x800) {
              s.push_back(static_cast<char>(0xC0 | (code >> 6)));
              s.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
              s.push_back(static_cast<char>(0xE0 | (code >> 12)));
              s.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              s.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            break;
          }
          default:
            return fail("invalid escape");
        }
      } else {
        s.push_back(c);
      }
    }
    return fail("unterminated string");
  }

  Error parse_array(Value* out) {
    if (!consume('[')) {
      return fail("expected '['");
    }
    Array arr;
    skip_ws();
    if (consume(']')) {
      *out = Value::array(std::move(arr));
      return Error::success();
    }
    while (true) {
      Value item;
      Error err = parse_value(&item);
      if (!err.ok()) {
        return err;
      }
      arr.push_back(std::move(item));
      skip_ws();
      if (consume(']')) {
        *out = Value::array(std::move(arr));
        return Error::success();
      }
      if (!consume(',')) {
        return fail("expected ',' or ']' in array");
      }
    }
  }

  Error parse_object(Value* out) {
    if (!consume('{')) {
      return fail("expected '{'");
    }
    Object obj;
    skip_ws();
    if (consume('}')) {
      *out = Value::object(std::move(obj));
      return Error::success();
    }
    while (true) {
      Value key_v;
      Error err = parse_value(&key_v);
      if (!err.ok()) {
        return err;
      }
      if (!key_v.is_string()) {
        return fail("object key must be string");
      }
      if (!consume(':')) {
        return fail("expected ':'");
      }
      Value val;
      err = parse_value(&val);
      if (!err.ok()) {
        return err;
      }
      obj[std::move(key_v.str)] = std::move(val);
      skip_ws();
      if (consume('}')) {
        *out = Value::object(std::move(obj));
        return Error::success();
      }
      if (!consume(',')) {
        return fail("expected ',' or '}' in object");
      }
    }
  }
};

void write_escaped(std::ostream& os, const std::string& s) {
  os << '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\b': os << "\\b"; break;
      case '\f': os << "\\f"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default:
        if (c < 0x20) {
          const char* hex = "0123456789abcdef";
          os << "\\u00" << hex[(c >> 4) & 0xF] << hex[c & 0xF];
        } else {
          os << static_cast<char>(c);
        }
    }
  }
  os << '"';
}

void dump(std::ostream& os, const Value& v, bool pretty, int indent) {
  auto pad = [&](int n) {
    if (pretty) {
      for (int i = 0; i < n; ++i) os << "  ";
    }
  };
  switch (v.kind) {
    case Value::Kind::Null:
      os << "null";
      break;
    case Value::Kind::Bool:
      os << (v.b ? "true" : "false");
      break;
    case Value::Kind::Number: {
      if (std::isfinite(v.number) && std::floor(v.number) == v.number &&
          std::abs(v.number) < 1e15) {
        os << static_cast<long long>(v.number);
      } else {
        os << v.number;
      }
      break;
    }
    case Value::Kind::String:
      write_escaped(os, v.str);
      break;
    case Value::Kind::Arr: {
      os << '[';
      if (pretty && !v.arr.empty()) os << '\n';
      for (std::size_t i = 0; i < v.arr.size(); ++i) {
        if (pretty) pad(indent + 1);
        dump(os, v.arr[i], pretty, indent + 1);
        if (i + 1 < v.arr.size()) os << ',';
        if (pretty) os << '\n';
      }
      if (pretty && !v.arr.empty()) pad(indent);
      os << ']';
      break;
    }
    case Value::Kind::Obj: {
      os << '{';
      if (pretty && !v.obj.empty()) os << '\n';
      std::size_t i = 0;
      for (const auto& kv : v.obj) {
        if (pretty) pad(indent + 1);
        write_escaped(os, kv.first);
        os << (pretty ? ": " : ":");
        dump(os, kv.second, pretty, indent + 1);
        if (i + 1 < v.obj.size()) os << ',';
        if (pretty) os << '\n';
        ++i;
      }
      if (pretty && !v.obj.empty()) pad(indent);
      os << '}';
      break;
    }
  }
}

}  // namespace

Error parse(const std::string& text, Value* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "json out is null");
  }
  Parser p(text);
  return p.parse(out);
}

std::string stringify(const Value& value, bool pretty) {
  std::ostringstream oss;
  dump(oss, value, pretty, 0);
  if (pretty) {
    oss << '\n';
  }
  return oss.str();
}

const Value* get(const Object& obj, const char* key) {
  const auto it = obj.find(key);
  if (it == obj.end()) {
    return nullptr;
  }
  return &it->second;
}

Error require_object(const Value& v, const Object** out, const char* ctx) {
  if (!v.is_object()) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string(ctx) + ": expected object");
  }
  *out = &v.obj;
  return Error::success();
}

Error require_array(const Value& v, const Array** out, const char* ctx) {
  if (!v.is_array()) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string(ctx) + ": expected array");
  }
  *out = &v.arr;
  return Error::success();
}

}  // namespace json
}  // namespace ir
}  // namespace uaii
