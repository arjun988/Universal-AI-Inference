#pragma once

#include "uaii/export.hpp"
#include "uaii/interfaces/storage.hpp"

#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace uaii {
namespace storage {

/// Disk-backed storage provider with optional mmap-style whole-file staging.
/// Cross-platform: uses buffered file IO (mmap semantics via tier hint).
class UAII_API FileStorageProvider : public IStorageProvider {
 public:
  explicit FileStorageProvider(bool prefer_mmap = true);

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
  bool prefer_mmap_ = true;
  std::uint64_t next_id_ = 1;
  std::uint64_t bytes_read_ = 0;
  mutable std::mutex mu_;
  std::unordered_map<std::uint64_t, std::string> paths_;
};

}  // namespace storage
}  // namespace uaii
