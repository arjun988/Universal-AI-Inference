#include "uaii/storage/streaming.hpp"

#include "uaii/core/log.hpp"

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
  resident_[0] = resident_[1] = 0;
  active_ = 0;
  prefetch_id_ = 0;
  weights_dir_ = weights_dir;
  buffers_[0].assign(static_cast<std::size_t>(plan.staging_bytes), 0);
  buffers_[1].assign(static_cast<std::size_t>(plan.staging_bytes), 0);

  for (const auto& p : plan.placements) {
    if (!p.stream) continue;
    const std::string path = join_path(weights_dir_, p.uri);
    TensorHandle h;
    Error err = provider_.open(path, &h);
    if (!err.ok()) {
      h.id = 0;
      h.uri = path;
      h.size_bytes = p.bytes;
      h.tier = StorageTier::Disk;
    }
    handles_[p.tensor_id] = h;
    sizes_[p.tensor_id] = p.bytes;
  }
  log::info("storage") << "streaming configured staging=" << buffers_[0].size()
                       << "B x2 tensors=" << handles_.size();
  return Error::success();
}

bool StreamingWeightStore::is_streamed(TensorId id) const noexcept {
  return sizes_.count(id) != 0;
}

Error StreamingWeightStore::stage_into(int slot, TensorId id) {
  auto sit = sizes_.find(id);
  if (sit == sizes_.end()) {
    return Error::make(ErrorCode::NotFound, "tensor not streamed");
  }
  if (sit->second > buffers_[slot].size()) {
    return Error::make(ErrorCode::InvalidArgument, "weight exceeds staging buffer");
  }
  auto hit = handles_.find(id);
  if (hit == handles_.end()) {
    return Error::make(ErrorCode::NotFound, "no handle for streamed weight");
  }
  TensorHandle& h = hit->second;
  if (h.id == 0) {
    Error err = provider_.open(h.uri, &h);
    if (!err.ok()) return err;
  }
  Error err = provider_.read(h, 0, sit->second, buffers_[slot].data());
  if (!err.ok()) return err;
  resident_[slot] = id;
  return Error::success();
}

void StreamingWeightStore::prefetch(TensorId id) {
  if (!is_streamed(id)) return;
  std::lock_guard<std::mutex> lock(mu_);
  if (resident_[active_] == id || resident_[1 - active_] == id) return;
  if (prefetch_fut_.valid()) return;  // one in flight
  const int slot = 1 - active_;
  prefetch_id_ = id;
  prefetch_fut_ = std::async(std::launch::async, [this, slot, id]() {
    return stage_into(slot, id);
  });
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
    active_ = 1 - active_;
  } else {
    if (prefetch_fut_.valid()) {
      (void)prefetch_fut_.get();
      prefetch_id_ = 0;
    }
    if (resident_[active_] != id) {
      if (resident_[1 - active_] == id) {
        active_ = 1 - active_;
      } else {
        Error err = stage_into(active_, id);
        if (!err.ok()) return err;
      }
    }
  }

  *out_data = buffers_[active_].data();
  *out_nbytes = static_cast<std::size_t>(sit->second);
  return Error::success();
}

void StreamingWeightStore::reset_stats() noexcept {
  provider_.reset_stats();
}

}  // namespace storage
}  // namespace uaii
