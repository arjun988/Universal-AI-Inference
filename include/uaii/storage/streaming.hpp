#pragma once

#include "uaii/export.hpp"
#include "uaii/planner/storage_plan.hpp"
#include "uaii/storage/file_provider.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace uaii {
namespace storage {

/// Streams oversized weights into a fixed staging buffer under a RAM budget.
class UAII_API StreamingWeightStore {
 public:
  StreamingWeightStore();

  [[nodiscard]] Error configure(const planner::StoragePlan& plan,
                                const std::string& weights_dir);

  [[nodiscard]] bool is_streamed(TensorId id) const noexcept;
  [[nodiscard]] std::uint64_t staging_bytes() const noexcept { return staging_.size(); }
  [[nodiscard]] std::uint64_t bytes_read() const noexcept { return provider_.bytes_read(); }
  [[nodiscard]] const FileStorageProvider& provider() const noexcept { return provider_; }

  /// Ensure weight `id` is resident in staging; returns pointer valid until next stage.
  [[nodiscard]] Error stage(TensorId id, const void** out_data, std::size_t* out_nbytes);

  void reset_stats() noexcept;

 private:
  FileStorageProvider provider_;
  std::string weights_dir_;
  std::vector<std::uint8_t> staging_;
  TensorId resident_id_ = 0;
  std::unordered_map<TensorId, TensorHandle> handles_;
  std::unordered_map<TensorId, std::uint64_t> sizes_;
};

}  // namespace storage
}  // namespace uaii
