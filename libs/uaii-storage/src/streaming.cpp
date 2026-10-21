#include "uaii/storage/streaming.hpp"

#include "uaii/core/log.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace uaii {
namespace storage {
namespace {

std::string join_path(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
#if defined(_WIN32)
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  if (!file.empty() && (file[0] == '/' || file[0] == '\\')) return file;
  std::string path = file;
  const auto hash = path.find('#');
  if (hash != std::string::npos) path = path.substr(0, hash);
  if (dir.empty()) return path;
  if (dir.back() == '/' || dir.back() == '\\') return dir + path;
  return dir + sep + path;
}

}  // namespace

StreamingWeightStore::StreamingWeightStore() : provider_(true) {}

Error StreamingWeightStore::configure(const planner::StoragePlan& plan,
                                      const std::string& weights_dir) {
  if (prefetch_fut_.valid()) {
    (void)prefetch_fut_.wait();
  }
  handles_.clear();
  sizes_.clear();
  expert_.clear();
  expert_store_.clear();
  expert_lru_.clear();
  expert_hot_.clear();
  expert_used_ = 0;
  expert_hits_ = 0;
  expert_misses_ = 0;
  loader_bytes_ = 0;
  resident_[0] = resident_[1] = 0;
  active_ = 0;
  prefetch_id_ = 0;
  weights_dir_ = weights_dir;
  expert_budget_ = plan.expert_cache_bytes;
  const std::size_t staging = static_cast<std::size_t>(
      std::max<std::uint64_t>(plan.staging_bytes, 1));
  const int slots = plan.staging_slots <= 1 ? 1 : 2;
  buffers_[0].assign(staging, 0);
  if (slots <= 1) {
    buffers_[1].clear();
  } else {
    buffers_[1].assign(staging, 0);
  }

  for (const auto& p : plan.placements) {
    if (!p.stream) continue;
    TensorHandle h;
    h.id = 0;
    h.uri = p.uri.empty() ? p.name : p.uri;
    h.size_bytes = p.bytes;
    h.tier = StorageTier::Disk;
    handles_[p.tensor_id] = h;
    sizes_[p.tensor_id] = p.bytes;
    expert_[p.tensor_id] = p.expert;
  }
  log::info("storage") << "streaming configured staging=" << buffers_[0].size()
                       << "B x" << slots << " tensors=" << handles_.size()
                       << " expert_budget=" << expert_budget_;
  return Error::success();
}

bool StreamingWeightStore::is_streamed(TensorId id) const noexcept {
  return sizes_.count(id) != 0;
}

bool StreamingWeightStore::is_expert(TensorId id) const noexcept {
  auto it = expert_.find(id);
  return it != expert_.end() && it->second;
}

void StreamingWeightStore::expert_touch(TensorId id) {
  expert_lru_.remove(id);
  expert_lru_.push_front(id);
}

bool StreamingWeightStore::expert_is_hot(TensorId id) const {
  return std::find(expert_hot_.begin(), expert_hot_.end(), id) != expert_hot_.end();
}

void StreamingWeightStore::expert_mark_hot(TensorId id) {
  expert_hot_.erase(std::remove(expert_hot_.begin(), expert_hot_.end(), id), expert_hot_.end());
  expert_hot_.push_back(id);
  constexpr std::size_t kMaxHot = 32;
  if (expert_hot_.size() > kMaxHot) {
    expert_hot_.erase(expert_hot_.begin(),
                      expert_hot_.begin() + static_cast<std::ptrdiff_t>(expert_hot_.size() - kMaxHot));
  }
}

void StreamingWeightStore::expert_evict_to_fit(std::size_t incoming) {
  if (expert_budget_ == 0) return;
  while (!expert_lru_.empty() &&
         expert_used_ + static_cast<std::uint64_t>(incoming) > expert_budget_) {
    TensorId victim = 0;
    for (auto it = expert_lru_.rbegin(); it != expert_lru_.rend(); ++it) {
      if (!expert_is_hot(*it)) {
        victim = *it;
        break;
      }
    }
    if (victim == 0) break;  // all hot — keep them; budget may be exceeded briefly
    expert_lru_.remove(victim);
    auto sit = expert_store_.find(victim);
    if (sit == expert_store_.end()) continue;
    if (expert_used_ >= sit->second.size()) {
      expert_used_ -= sit->second.size();
    } else {
      expert_used_ = 0;
    }
    expert_store_.erase(sit);
  }
}

void StreamingWeightStore::expert_put(TensorId id, const void* data, std::size_t nbytes) {
  if (data == nullptr || nbytes == 0) return;
  if (expert_budget_ == 0) {
    expert_budget_ = std::max<std::uint64_t>(static_cast<std::uint64_t>(nbytes) * 4ull,
                                            64ull * 1024ull * 1024ull);
  }
  expert_evict_to_fit(nbytes);
  std::vector<std::uint8_t> buf(nbytes);
  std::memcpy(buf.data(), data, nbytes);
  auto it = expert_store_.find(id);
  if (it != expert_store_.end()) {
    if (expert_used_ >= it->second.size()) expert_used_ -= it->second.size();
    it->second = std::move(buf);
  } else {
    expert_store_[id] = std::move(buf);
  }
  expert_used_ += static_cast<std::uint64_t>(nbytes);
  expert_touch(id);
  expert_mark_hot(id);
}

bool StreamingWeightStore::expert_get(TensorId id, void* dst, std::size_t nbytes) {
  auto it = expert_store_.find(id);
  if (it == expert_store_.end() || it->second.size() != nbytes || dst == nullptr) {
    return false;
  }
  std::memcpy(dst, it->second.data(), nbytes);
  expert_touch(id);
  return true;
}

Error StreamingWeightStore::stage_into(int slot, TensorId id) {
  auto sit = sizes_.find(id);
  if (sit == sizes_.end()) {
    return Error::make(ErrorCode::NotFound, "tensor not streamed");
  }
  if (sit->second > buffers_[slot].size()) {
    return Error::make(ErrorCode::InvalidArgument, "weight exceeds staging buffer");
  }
  const std::size_t n = static_cast<std::size_t>(sit->second);
  if (is_expert(id) && expert_get(id, buffers_[slot].data(), n)) {
    ++expert_hits_;
    resident_[slot] = id;
    return Error::success();
  }
  if (is_expert(id)) ++expert_misses_;

  auto hit = handles_.find(id);
  if (hit == handles_.end()) {
    return Error::make(ErrorCode::NotFound, "no handle for streamed weight");
  }
  if (loader_) {
    Error err = loader_(id, hit->second.uri, buffers_[slot].data(), n);
    if (!err.ok()) return err;
    loader_bytes_ += n;
  } else {
    TensorHandle& h = hit->second;
    if (h.id == 0) {
      const std::string path = join_path(weights_dir_, h.uri);
      Error err = provider_.open(path, &h);
      if (!err.ok()) return err;
    }
    Error err = provider_.read(h, 0, sit->second, buffers_[slot].data());
    if (!err.ok()) return err;
  }
  if (is_expert(id)) {
    expert_put(id, buffers_[slot].data(), n);
  }
  resident_[slot] = id;
  return Error::success();
}

void StreamingWeightStore::prefetch(TensorId id) {
  if (!is_streamed(id)) return;
  if (buffers_[1].empty()) return;
  std::lock_guard<std::mutex> lock(mu_);
  if (resident_[active_] == id || resident_[1 - active_] == id) return;
  if (prefetch_fut_.valid()) return;  // one in flight
  const int slot = 1 - active_;
  prefetch_id_ = id;
  prefetch_fut_ = std::async(std::launch::async, [this, slot, id]() {
    return stage_into(slot, id);
  });
}

TensorId StreamingWeightStore::slot_resident(int slot) const noexcept {
  if (slot < 0 || slot > 1) return 0;
  return resident_[slot];
}

const void* StreamingWeightStore::slot_data(int slot) const noexcept {
  if (slot < 0 || slot > 1) return nullptr;
  return buffers_[slot].data();
}

std::size_t StreamingWeightStore::tensor_bytes(TensorId id) const {
  auto it = sizes_.find(id);
  if (it == sizes_.end()) return 0;
  return static_cast<std::size_t>(it->second);
}

bool StreamingWeightStore::try_peek_prefetched(TensorId id, const void** out_data,
                                               std::size_t* out_nbytes) {
  if (out_data == nullptr || out_nbytes == nullptr) return false;
  std::lock_guard<std::mutex> lock(mu_);
  if (prefetch_fut_.valid() && prefetch_id_ == id) {
    if (prefetch_fut_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      return false;
    }
  }
  const int inactive = 1 - active_;
  if (buffers_[inactive].empty()) return false;
  if (resident_[inactive] != id) return false;
  auto sit = sizes_.find(id);
  if (sit == sizes_.end()) return false;
  *out_data = buffers_[inactive].data();
  *out_nbytes = static_cast<std::size_t>(sit->second);
  return true;
}

Error StreamingWeightStore::stage(TensorId id, const void** out_data, std::size_t* out_nbytes) {
  if (out_data == nullptr || out_nbytes == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "stage out null");
  }
  auto sit = sizes_.find(id);
  if (sit == sizes_.end()) {
    return Error::make(ErrorCode::NotFound, "tensor not streamed");
  }

  std::lock_guard<std::mutex> lock(mu_);
  if (prefetch_fut_.valid() && prefetch_id_ == id) {
    Error err = prefetch_fut_.get();
    prefetch_id_ = 0;
    if (!err.ok()) return err;
    const int other = 1 - active_;
    if (!buffers_[other].empty()) active_ = other;
  } else {
    if (prefetch_fut_.valid()) {
      (void)prefetch_fut_.get();
      prefetch_id_ = 0;
    }
    if (resident_[active_] != id) {
      const int other = 1 - active_;
      if (!buffers_[other].empty() && resident_[other] == id) {
        active_ = other;
      } else {
        Error err = stage_into(active_, id);
        if (!err.ok()) return err;
      }
    }
  }

  if (is_expert(id)) {
    auto eit = expert_store_.find(id);
    if (eit != expert_store_.end() && !eit->second.empty()) {
      expert_mark_hot(id);
      *out_data = eit->second.data();
      *out_nbytes = eit->second.size();
      return Error::success();
    }
  }
  *out_data = buffers_[active_].data();
  *out_nbytes = static_cast<std::size_t>(sit->second);
  return Error::success();
}

void StreamingWeightStore::reset_stats() noexcept {
  provider_.reset_stats();
  loader_bytes_ = 0;
  expert_hits_ = 0;
  expert_misses_ = 0;
}

}  // namespace storage
}  // namespace uaii
