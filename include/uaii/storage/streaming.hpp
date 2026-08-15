#pragma once

#include "uaii/export.hpp"
#include "uaii/planner/storage_plan.hpp"
#include "uaii/storage/file_provider.hpp"

#include <functional>
#include <future>
#include <list>
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

  /// Optional: load `file#tensor` GGUF/safetensors refs (session installs this).
  using TensorLoader = std::function<Error(TensorId id, const std::string& weight_ref,
                                           void* dst, std::size_t nbytes)>;
  void set_tensor_loader(TensorLoader loader) { loader_ = std::move(loader); }

  [[nodiscard]] bool is_streamed(TensorId id) const noexcept;
  [[nodiscard]] bool is_expert(TensorId id) const noexcept;
  [[nodiscard]] std::uint64_t staging_bytes() const noexcept {
    return buffers_[0].size();
  }
  [[nodiscard]] std::uint64_t bytes_read() const noexcept { return provider_.bytes_read() + loader_bytes_; }
  [[nodiscard]] std::uint64_t expert_cache_hits() const noexcept { return expert_hits_; }
  [[nodiscard]] std::uint64_t expert_cache_misses() const noexcept { return expert_misses_; }
  [[nodiscard]] std::uint64_t expert_cache_used_bytes() const noexcept { return expert_used_; }
  [[nodiscard]] const FileStorageProvider& provider() const noexcept { return provider_; }

  /// Ensure weight `id` is resident; pointer valid until next stage of same slot.
  [[nodiscard]] Error stage(TensorId id, const void** out_data, std::size_t* out_nbytes);

  /// Kick off async load of `id` into the inactive buffer (best-effort).
  void prefetch(TensorId id);

  /// Host staging pointer for `slot` (0 or 1) when `slot_resident(slot) != 0`.
  [[nodiscard]] TensorId slot_resident(int slot) const noexcept;
  [[nodiscard]] const void* slot_data(int slot) const noexcept;
  [[nodiscard]] std::size_t tensor_bytes(TensorId id) const;

  /// If `id` is ready in the inactive prefetch slot, return host bytes (non-consuming).
  [[nodiscard]] bool try_peek_prefetched(TensorId id, const void** out_data,
                                         std::size_t* out_nbytes);

  void reset_stats() noexcept;

 private:
  [[nodiscard]] Error stage_into(int slot, TensorId id);

  FileStorageProvider provider_;
  std::string weights_dir_;
  TensorLoader loader_;
  std::vector<std::uint8_t> buffers_[2];
  TensorId resident_[2] = {0, 0};
  int active_ = 0;
  std::mutex mu_;
  std::future<Error> prefetch_fut_;
  TensorId prefetch_id_ = 0;
  std::unordered_map<TensorId, TensorHandle> handles_;
  std::unordered_map<TensorId, std::uint64_t> sizes_;
  std::unordered_map<TensorId, bool> expert_;
  std::uint64_t expert_budget_ = 0;
  std::uint64_t expert_used_ = 0;
  std::uint64_t expert_hits_ = 0;
  std::uint64_t expert_misses_ = 0;
  std::uint64_t loader_bytes_ = 0;
  std::list<TensorId> expert_lru_;
  std::unordered_map<TensorId, std::vector<std::uint8_t>> expert_store_;
  std::vector<TensorId> expert_hot_;

  void expert_touch(TensorId id);
  void expert_evict_to_fit(std::size_t incoming);
  void expert_put(TensorId id, const void* data, std::size_t nbytes);
  [[nodiscard]] bool expert_get(TensorId id, void* dst, std::size_t nbytes);
  [[nodiscard]] bool expert_is_hot(TensorId id) const;
  void expert_mark_hot(TensorId id);
};

}  // namespace storage
}  // namespace uaii
