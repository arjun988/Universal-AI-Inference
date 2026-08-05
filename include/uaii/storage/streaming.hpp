#pragma once

#include "uaii/export.hpp"
#include "uaii/planner/storage_plan.hpp"
#include "uaii/storage/file_provider.hpp"

#include <atomic>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace storage {

/// Streams oversized weights into double-buffered staging (prefetch + compute).
class UAII_API StreamingWeightStore {
 public:
  StreamingWeightStore();

  [[nodiscard]] Error configure(const planner::StoragePlan& plan,
                                const std::string& weights_dir);

  [[nodiscard]] bool is_streamed(TensorId id) const noexcept;
  [[nodiscard]] std::uint64_t staging_bytes() const noexcept {
    return buffers_[0].size();
  }
  [[nodiscard]] std::uint64_t bytes_read() const noexcept { return provider_.bytes_read(); }
  [[nodiscard]] const FileStorageProvider& provider() const noexcept { return provider_; }

  /// Ensure weight `id` is resident; pointer valid until next stage of same slot.
  [[nodiscard]] Error stage(TensorId id, const void** out_data, std::size_t* out_nbytes);

  /// Kick off async load of `id` into the inactive buffer (best-effort).
  void prefetch(TensorId id);

  void reset_stats() noexcept;

 private:
  [[nodiscard]] Error stage_into(int slot, TensorId id);

  FileStorageProvider provider_;
  std::string weights_dir_;
  std::vector<std::uint8_t> buffers_[2];
  TensorId resident_[2] = {0, 0};
  int active_ = 0;
  std::mutex mu_;
  std::future<Error> prefetch_fut_;
  TensorId prefetch_id_ = 0;
  std::unordered_map<TensorId, TensorHandle> handles_;
  std::unordered_map<TensorId, std::uint64_t> sizes_;
};

}  // namespace storage
}  // namespace uaii
