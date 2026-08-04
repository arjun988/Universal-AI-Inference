#pragma once

#include "uaii/core/error.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace uaii {
namespace loaders {
namespace detail {

class ByteReader {
 public:
  explicit ByteReader(std::vector<std::uint8_t> data) : data_(std::move(data)) {}

  [[nodiscard]] static Error from_file(const std::string& path, ByteReader* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return Error::make(ErrorCode::NotFound, "cannot open " + path);
    }
    in.seekg(0, std::ios::end);
    const auto sz = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> buf(sz);
    if (sz && !in.read(reinterpret_cast<char*>(buf.data()),
                       static_cast<std::streamsize>(sz))) {
      return Error::make(ErrorCode::IoError, "failed reading " + path);
    }
    *out = ByteReader(std::move(buf));
    return Error::ok();
  }

  [[nodiscard]] std::size_t tell() const noexcept { return pos_; }
  void seek(std::size_t p) { pos_ = p; }
  [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
  [[nodiscard]] const std::uint8_t* data() const noexcept { return data_.data(); }

  template <typename T>
  Error read_pod(T* out) {
    if (pos_ + sizeof(T) > data_.size()) {
      return Error::make(ErrorCode::InvalidArgument, "truncated binary read");
    }
    std::memcpy(out, data_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return Error::ok();
  }

  Error read_bytes(void* dst, std::size_t n) {
    if (pos_ + n > data_.size()) {
      return Error::make(ErrorCode::InvalidArgument, "truncated binary read");
    }
    std::memcpy(dst, data_.data() + pos_, n);
    pos_ += n;
    return Error::ok();
  }

  Error read_string(std::string* out) {
    std::uint64_t n = 0;
    Error err = read_pod(&n);
    if (!err.ok()) return err;
    if (pos_ + static_cast<std::size_t>(n) > data_.size()) {
      return Error::make(ErrorCode::InvalidArgument, "truncated string");
    }
    out->assign(reinterpret_cast<const char*>(data_.data() + pos_),
                static_cast<std::size_t>(n));
    pos_ += static_cast<std::size_t>(n);
    return Error::ok();
  }

 private:
  std::vector<std::uint8_t> data_;
  std::size_t pos_ = 0;
};

class ByteWriter {
 public:
  void write_pod(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    data_.insert(data_.end(), b, b + n);
  }

  template <typename T>
  void write_pod(const T& v) {
    write_pod(&v, sizeof(T));
  }

  void write_string(const std::string& s) {
    const std::uint64_t n = s.size();
    write_pod(n);
    if (!s.empty()) {
      write_pod(s.data(), s.size());
    }
  }

  void pad_to(std::size_t alignment) {
    while (data_.size() % alignment != 0) {
      data_.push_back(0);
    }
  }

  [[nodiscard]] Error save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      return Error::make(ErrorCode::IoError, "cannot write " + path);
    }
    if (!data_.empty()) {
      out.write(reinterpret_cast<const char*>(data_.data()),
                static_cast<std::streamsize>(data_.size()));
    }
    if (!out) {
      return Error::make(ErrorCode::IoError, "failed writing " + path);
    }
    return Error::ok();
  }

  [[nodiscard]] std::vector<std::uint8_t>& bytes() { return data_; }
  [[nodiscard]] std::size_t size() const { return data_.size(); }

 private:
  std::vector<std::uint8_t> data_;
};

inline std::string lower_ext(const std::string& path) {
  auto pos = path.find_last_of('.');
  if (pos == std::string::npos) return {};
  std::string e = path.substr(pos);
  for (char& c : e) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return e;
}

}  // namespace detail
}  // namespace loaders
}  // namespace uaii
