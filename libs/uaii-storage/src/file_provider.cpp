#include "uaii/storage/file_provider.hpp"

#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace uaii {
namespace storage {
namespace {

struct MappedFile {
  std::string path;
  std::uint64_t size = 0;
  void* view = nullptr;
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
#else
  int fd = -1;
#endif
};

}  // namespace

struct FileStorageProvider::Impl {
  std::unordered_map<std::uint64_t, MappedFile> maps;
};

FileStorageProvider::FileStorageProvider(bool prefer_mmap)
    : prefer_mmap_(prefer_mmap), impl_(std::make_unique<Impl>()) {}

FileStorageProvider::~FileStorageProvider() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!impl_) return;
  for (auto& kv : impl_->maps) {
#if defined(_WIN32)
    if (kv.second.view) UnmapViewOfFile(kv.second.view);
    if (kv.second.mapping) CloseHandle(kv.second.mapping);
    if (kv.second.file != INVALID_HANDLE_VALUE) CloseHandle(kv.second.file);
#else
    if (kv.second.view && kv.second.size)
      munmap(kv.second.view, static_cast<size_t>(kv.second.size));
    if (kv.second.fd >= 0) ::close(kv.second.fd);
#endif
  }
  impl_->maps.clear();
}
 
Error FileStorageProvider::open(const std::string& uri, TensorHandle* out) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "handle out null");
  }

  if (prefer_mmap_) {
    MappedFile mf;
    mf.path = uri;
#if defined(_WIN32)
    {
      int n = MultiByteToWideChar(CP_UTF8, 0, uri.c_str(), -1, nullptr, 0);
      std::wstring wuri;
      if (n > 0) {
        wuri.resize(static_cast<std::size_t>(n));
        MultiByteToWideChar(CP_UTF8, 0, uri.c_str(), -1, wuri.data(), n);
        mf.file = CreateFileW(wuri.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      } else {
        mf.file = CreateFileA(uri.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      }
    }
    if (mf.file == INVALID_HANDLE_VALUE) {
      return Error::make(ErrorCode::IoError, "CreateFile failed " + uri);
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(mf.file, &sz)) {
      CloseHandle(mf.file);
      return Error::make(ErrorCode::IoError, "GetFileSizeEx failed");
    }
    mf.size = static_cast<std::uint64_t>(sz.QuadPart);
    if (mf.size == 0) {
      CloseHandle(mf.file);
      return Error::make(ErrorCode::IoError, "empty file " + uri);
    }
    mf.mapping = CreateFileMappingA(mf.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mf.mapping) {
      CloseHandle(mf.file);
      return Error::make(ErrorCode::IoError, "CreateFileMapping failed");
    }
    mf.view = MapViewOfFile(mf.mapping, FILE_MAP_READ, 0, 0, 0);
    if (!mf.view) {
      CloseHandle(mf.mapping);
      CloseHandle(mf.file);
      return Error::make(ErrorCode::IoError, "MapViewOfFile failed");
    }
#else
    mf.fd = ::open(uri.c_str(), O_RDONLY);
    if (mf.fd < 0) {
      return Error::make(ErrorCode::IoError, "open failed " + uri);
    }
    struct stat st {};
    if (fstat(mf.fd, &st) != 0) {
      ::close(mf.fd);
      return Error::make(ErrorCode::IoError, "fstat failed");
    }
    mf.size = static_cast<std::uint64_t>(st.st_size);
    if (mf.size == 0) {
      ::close(mf.fd);
      return Error::make(ErrorCode::IoError, "empty file " + uri);
    }
    mf.view = mmap(nullptr, static_cast<size_t>(mf.size), PROT_READ, MAP_PRIVATE, mf.fd, 0);
    if (mf.view == MAP_FAILED) {
      ::close(mf.fd);
      mf.view = nullptr;
      return Error::make(ErrorCode::IoError, "mmap failed " + uri);
    }
#endif
    std::lock_guard<std::mutex> lock(mu_);
    out->id = next_id_++;
    out->tier = StorageTier::Mmap;
    out->uri = uri;
    out->offset = 0;
    out->size_bytes = mf.size;
    paths_[out->id] = uri;
    impl_->maps.emplace(out->id, std::move(mf));
    return Error::success();
  }

  // Buffered fallback
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
  out->tier = StorageTier::Disk;
  out->uri = uri;
  out->offset = 0;
  out->size_bytes = static_cast<std::uint64_t>(sz);
  paths_[out->id] = uri;
  return Error::success();
}

Error FileStorageProvider::close(const TensorHandle& handle) {
  std::lock_guard<std::mutex> lock(mu_);
  paths_.erase(handle.id);
  if (impl_) {
    auto it = impl_->maps.find(handle.id);
    if (it != impl_->maps.end()) {
#if defined(_WIN32)
      if (it->second.view) UnmapViewOfFile(it->second.view);
      if (it->second.mapping) CloseHandle(it->second.mapping);
      if (it->second.file != INVALID_HANDLE_VALUE) CloseHandle(it->second.file);
#else
      if (it->second.view && it->second.size)
        munmap(it->second.view, static_cast<size_t>(it->second.size));
      if (it->second.fd >= 0) ::close(it->second.fd);
#endif
      impl_->maps.erase(it);
    }
  }
  return Error::success();
}

Error FileStorageProvider::read(const TensorHandle& handle,
                                std::uint64_t offset,
                                std::uint64_t size,
                                void* dst) {
  if (dst == nullptr || size == 0) {
    return Error::make(ErrorCode::InvalidArgument, "read dst/size invalid");
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    if (impl_) {
      auto it = impl_->maps.find(handle.id);
      if (it != impl_->maps.end() && it->second.view != nullptr) {
        if (offset + size > it->second.size) {
          return Error::make(ErrorCode::InvalidArgument, "mmap read OOB");
        }
        std::memcpy(dst,
                    static_cast<const char*>(it->second.view) +
                        static_cast<std::size_t>(handle.offset + offset),
                    static_cast<std::size_t>(size));
        bytes_read_ += size;
        return Error::success();
      }
    }
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
  return Error::success();
}

}  // namespace storage
}  // namespace uaii
