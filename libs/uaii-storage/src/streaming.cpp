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
  // weight_ref may be "path#tensor" — strip fragment for file open
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
  handles_.clear();
  sizes_.clear();
  resident_id_ = 0;
  weights_dir_ = weights_dir;
  staging_.assign(static_cast<std::size_t>(plan.staging_bytes), 0);

  for (const auto& p : plan.placements) {
    if (!p.stream) continue;
    const std::string path = join_path(weights_dir_, p.uri);
    TensorHandle h;
    Error err = provider_.open(path, &h);
    if (!err.ok()) {
      // Allow configure to succeed for demo fixtures created later; retry on stage.
      h.id = 0;
      h.uri = path;
      h.size_bytes = p.bytes;
      h.tier = StorageTier::Disk;
    }
    handles_[p.tensor_id] = h;
    sizes_[p.tensor_id] = p.bytes;
  }
  log::info("storage") << "streaming configured staging=" << staging_.size()
                       << "B tensors=" << handles_.size();
  return Error::ok();
}

bool StreamingWeightStore::is_streamed(TensorId id) const noexcept {
  return sizes_.count(id) != 0;
}

Error StreamingWeightStore::stage(TensorId id, const void** out_data, std::size_t* out_nbytes) {
  if (out_data == nullptr || out_nbytes == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "stage out null");
  }
  auto sit = sizes_.find(id);
  if (sit == sizes_.end()) {
    return Error::make(ErrorCode::NotFound, "tensor not streamed");
  }
  if (resident_id_ == id && !staging_.empty()) {
    *out_data = staging_.data();
    *out_nbytes = static_cast<std::size_t>(sit->second);
    return Error::ok();
  }
  if (sit->second > staging_.size()) {
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
  Error err = provider_.read(h, 0, sit->second, staging_.data());
  if (!err.ok()) return err;
  resident_id_ = id;
  *out_data = staging_.data();
  *out_nbytes = static_cast<std::size_t>(sit->second);
  return Error::ok();
}

void StreamingWeightStore::reset_stats() noexcept {
  provider_.reset_stats();
}

}  // namespace storage
}  // namespace uaii
