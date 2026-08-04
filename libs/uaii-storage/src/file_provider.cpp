#include "uaii/storage/file_provider.hpp"

#include <cstdio>

namespace uaii {
namespace storage {

FileStorageProvider::FileStorageProvider(bool prefer_mmap) : prefer_mmap_(prefer_mmap) {}

Error FileStorageProvider::open(const std::string& uri, TensorHandle* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "handle out null");
  }
  std::FILE* f = nullptr;
#if defined(_MSC_VER)
  if (fopen_s(&f, uri.c_str(), "rb") != 0) f = nullptr;
#else
  f = std::fopen(uri.c_str(), "rb");
#endif
  if (f == nullptr) {
    return Error::make(ErrorCode::IoError, "failed to open " + uri);
  }
  if (std::fseek(f, 0, SEEK_END) != 0) {
    std::fclose(f);
    return Error::make(ErrorCode::IoError, "seek failed " + uri);
  }
  const long sz = std::ftell(f);
  std::fclose(f);
  if (sz < 0) {
    return Error::make(ErrorCode::IoError, "ftell failed " + uri);
  }

  std::lock_guard<std::mutex> lock(mu_);
  out->id = next_id_++;
  out->tier = prefer_mmap_ ? StorageTier::Mmap : StorageTier::Disk;
  out->uri = uri;
  out->offset = 0;
  out->size_bytes = static_cast<std::uint64_t>(sz);
  paths_[out->id] = uri;
  return Error::ok();
}

Error FileStorageProvider::close(const TensorHandle& handle) {
  std::lock_guard<std::mutex> lock(mu_);
  paths_.erase(handle.id);
  return Error::ok();
}

Error FileStorageProvider::read(const TensorHandle& handle,
                                std::uint64_t offset,
                                std::uint64_t size,
                                void* dst) {
  if (dst == nullptr || size == 0) {
    return Error::make(ErrorCode::InvalidArgument, "read dst/size invalid");
  }
  std::string path;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = paths_.find(handle.id);
    if (it == paths_.end()) {
      return Error::make(ErrorCode::NotFound, "handle not open");
    }
    path = it->second;
  }

  std::FILE* f = nullptr;
#if defined(_MSC_VER)
  if (fopen_s(&f, path.c_str(), "rb") != 0) f = nullptr;
#else
  f = std::fopen(path.c_str(), "rb");
#endif
  if (f == nullptr) {
    return Error::make(ErrorCode::IoError, "failed to open for read " + path);
  }
  if (std::fseek(f, static_cast<long>(handle.offset + offset), SEEK_SET) != 0) {
    std::fclose(f);
    return Error::make(ErrorCode::IoError, "seek failed");
  }
  const std::size_t n = std::fread(dst, 1, static_cast<std::size_t>(size), f);
  std::fclose(f);
  if (n != static_cast<std::size_t>(size)) {
    return Error::make(ErrorCode::IoError, "short read from " + path);
  }
  bytes_read_ += size;
  return Error::ok();
}

}  // namespace storage
}  // namespace uaii
