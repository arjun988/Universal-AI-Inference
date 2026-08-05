#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/storage.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace uaii {
namespace storage {

/// Disk-backed storage with real OS mmap (Windows MapViewOfFile / POSIX mmap)
/// when prefer_mmap=true; otherwise buffered fread.
class UAII_API FileStorageProvider : public IStorageProvider {
 public:
  explicit FileStorageProvider(bool prefer_mmap = true);
  ~FileStorageProvider() override;

  FileStorageProvider(const FileStorageProvider&) = delete;
  FileStorageProvider& operator=(const FileStorageProvider&) = delete;

  [[nodiscard]] std::string name() const override { return "file"; }
  [[nodiscard]] StorageTier tier() const override {
    return prefer_mmap_ ? StorageTier::Mmap : StorageTier::Disk;
  }

  [[nodiscard]] Error open(const std::string& uri, TensorHandle* out) override;
  [[nodiscard]] Error close(const TensorHandle& handle) override;
  [[nodiscard]] Error read(const TensorHandle& handle,
                           std::uint64_t offset,
                           std::uint64_t size,
                           void* dst) override;

  [[nodiscard]] std::uint64_t bytes_read() const noexcept { return bytes_read_; }
  void reset_stats() noexcept { bytes_read_ = 0; }

 private:
  struct Impl;
  bool prefer_mmap_ = true;
  std::uint64_t next_id_ = 1;
  std::uint64_t bytes_read_ = 0;
  mutable std::mutex mu_;
  std::unordered_map<std::uint64_t, std::string> paths_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace storage
}  // namespace uaii
