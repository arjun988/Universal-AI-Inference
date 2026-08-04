#pragma once

#include "uaii/core/error.hpp"

#include <cstdint>
#include <string>

namespace uaii {

enum class StorageTier {
  Unknown = 0,
  Ram,
  Mmap,
  Disk,
  Nvme,
  ObjectStore,
  Remote,
  Compressed,
};

/// Location-agnostic tensor reference (storage-first design).
struct TensorHandle {
  std::uint64_t id = 0;
  StorageTier tier = StorageTier::Unknown;
  std::string uri;
  std::uint64_t offset = 0;
  std::uint64_t size_bytes = 0;
};

class IStorageProvider {
 public:
  virtual ~IStorageProvider() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual StorageTier tier() const = 0;

  [[nodiscard]] virtual Error open(const std::string& uri, TensorHandle* out) = 0;
  [[nodiscard]] virtual Error close(const TensorHandle& handle) = 0;

  /// Stage bytes into host memory (Phase 3/6 expand residency APIs).
  [[nodiscard]] virtual Error read(const TensorHandle& handle,
                                   std::uint64_t offset,
                                   std::uint64_t size,
                                   void* dst) = 0;
};

[[nodiscard]] inline const char* to_string(StorageTier tier) noexcept {
  switch (tier) {
    case StorageTier::Ram: return "ram";
    case StorageTier::Mmap: return "mmap";
    case StorageTier::Disk: return "disk";
    case StorageTier::Nvme: return "nvme";
    case StorageTier::ObjectStore: return "object_store";
    case StorageTier::Remote: return "remote";
    case StorageTier::Compressed: return "compressed";
    default: return "unknown";
  }
}

}  // namespace uaii
