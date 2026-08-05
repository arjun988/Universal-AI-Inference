#include "native_stubs.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__has_include)
#if __has_include(<webgpu/webgpu.h>)
#define UAII_HAVE_WEBGPU_HEADERS 1
#include <webgpu/webgpu.h>
#endif
#endif

namespace uaii {
namespace backends {
namespace native {
namespace {

#if defined(UAII_HAVE_WEBGPU_HEADERS)
std::mutex g_mu;
bool g_inited = false;
WGPUInstance g_instance = nullptr;
WGPUAdapter g_adapter = nullptr;
WGPUDevice g_device = nullptr;
WGPUQueue g_queue = nullptr;

struct BufferRec {
  WGPUBuffer buffer = nullptr;
  void* mapped = nullptr;
  std::size_t bytes = 0;
};
std::unordered_map<void*, BufferRec> g_bufs;

struct SyncUserData {
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPURequestAdapterStatus adapter_status = WGPURequestAdapterStatus_Error;
  WGPURequestDeviceStatus device_status = WGPURequestDeviceStatus_Error;
  bool adapter_done = false;
  bool device_done = false;
};

void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView /*message*/,
                void* userdata1, void* /*userdata2*/) {
  auto* ud = static_cast<SyncUserData*>(userdata1);
  ud->adapter_status = status;
  ud->adapter = adapter;
  ud->adapter_done = true;
}

void on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView /*message*/,
               void* userdata1, void* /*userdata2*/) {
  auto* ud = static_cast<SyncUserData*>(userdata1);
  ud->device_status = status;
  ud->device = device;
  ud->device_done = true;
}

Error wait_instance_events(WGPUInstance instance, SyncUserData* ud) {
  constexpr int kMaxPolls = 100000;
  for (int i = 0; i < kMaxPolls; ++i) {
    if (ud->adapter_done && ud->device_done) {
      break;
    }
    wgpuInstanceProcessEvents(instance);
  }
  if (!ud->adapter_done) {
    return Error::make(ErrorCode::Internal, "WebGPU adapter request timed out");
  }
  if (ud->adapter_status != WGPURequestAdapterStatus_Success || ud->adapter == nullptr) {
    return Error::make(ErrorCode::NotFound, "WebGPU adapter unavailable");
  }
  if (!ud->device_done) {
    return Error::make(ErrorCode::Internal, "WebGPU device request timed out");
  }
  if (ud->device_status != WGPURequestDeviceStatus_Success || ud->device == nullptr) {
    return Error::make(ErrorCode::Internal, "WebGPU device request failed");
  }
  return Error::success();
}
#endif

}  // namespace

bool webgpu_compiled() noexcept { return true; }

const char* webgpu_capability_details() noexcept {
#if defined(UAII_HAVE_WEBGPU_HEADERS)
  if (g_inited) {
    return "WebGPU native: buffer alloc/copy (MapWrite); compute dispatch pending; "
           "host_fallback: MatMul,Add,RMSNorm,Softmax,Silu,Attention,others";
  }
  return "WebGPU headers present; device acquire pending — host_fallback: "
         "MatMul,Add,RMSNorm,Softmax,Silu,Attention,others";
#else
  return "WebGPU headers unavailable — host_fallback: all ops";
#endif
}

Error webgpu_init() {
#if !defined(UAII_HAVE_WEBGPU_HEADERS)
  return Error::make(ErrorCode::NotImplemented,
                     "WebGPU headers not found (webgpu/webgpu.h)");
#else
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_inited) {
    return Error::success();
  }

  g_instance = wgpuCreateInstance(nullptr);
  if (g_instance == nullptr) {
    return Error::make(ErrorCode::Internal, "wgpuCreateInstance failed");
  }

  SyncUserData ud{};
  WGPURequestAdapterOptions aopts{};
  aopts.powerPreference = WGPUPowerPreference_HighPerformance;
  WGPURequestAdapterCallbackInfo acb{};
  acb.mode = WGPUCallbackMode_AllowProcessEvents;
  acb.callback = on_adapter;
  acb.userdata1 = &ud;
  wgpuInstanceRequestAdapter(g_instance, &aopts, acb);

  Error err = wait_instance_events(g_instance, &ud);
  if (!err.ok()) {
    wgpuInstanceRelease(g_instance);
    g_instance = nullptr;
    return err;
  }

  g_adapter = ud.adapter;
  WGPUDeviceDescriptor ddesc{};
  WGPURequestDeviceCallbackInfo dcb{};
  dcb.mode = WGPUCallbackMode_AllowProcessEvents;
  dcb.callback = on_device;
  dcb.userdata1 = &ud;
  ud.device_done = false;
  wgpuAdapterRequestDevice(g_adapter, &ddesc, dcb);
  err = wait_instance_events(g_instance, &ud);
  if (!err.ok()) {
    wgpuAdapterRelease(g_adapter);
    g_adapter = nullptr;
    wgpuInstanceRelease(g_instance);
    g_instance = nullptr;
    return err;
  }

  g_device = ud.device;
  g_queue = wgpuDeviceGetQueue(g_device);
  if (g_queue == nullptr) {
    wgpuDeviceRelease(g_device);
    g_device = nullptr;
    wgpuAdapterRelease(g_adapter);
    g_adapter = nullptr;
    wgpuInstanceRelease(g_instance);
    g_instance = nullptr;
    return Error::make(ErrorCode::Internal, "WebGPU queue unavailable");
  }

  g_inited = true;
  return Error::success();
#endif
}

void webgpu_shutdown() noexcept {
#if defined(UAII_HAVE_WEBGPU_HEADERS)
  std::lock_guard<std::mutex> lock(g_mu);
  for (auto& kv : g_bufs) {
    if (kv.second.buffer != nullptr) {
      wgpuBufferUnmap(kv.second.buffer);
      wgpuBufferDestroy(kv.second.buffer);
      wgpuBufferRelease(kv.second.buffer);
    }
  }
  g_bufs.clear();
  if (g_queue != nullptr) {
    wgpuQueueRelease(g_queue);
    g_queue = nullptr;
  }
  if (g_device != nullptr) {
    wgpuDeviceRelease(g_device);
    g_device = nullptr;
  }
  if (g_adapter != nullptr) {
    wgpuAdapterRelease(g_adapter);
    g_adapter = nullptr;
  }
  if (g_instance != nullptr) {
    wgpuInstanceRelease(g_instance);
    g_instance = nullptr;
  }
  g_inited = false;
#endif
}

Error webgpu_allocate(std::size_t bytes, void** out_ptr) {
#if !defined(UAII_HAVE_WEBGPU_HEADERS)
  (void)bytes;
  (void)out_ptr;
  return webgpu_init();
#else
  if (out_ptr == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "webgpu_allocate out null");
  }
  if (!g_inited || g_device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "WebGPU not initialized");
  }
  const std::size_t len = bytes == 0 ? 1 : bytes;
  WGPUBufferDescriptor desc{};
  desc.size = len;
  desc.usage = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
               WGPUBufferUsage_Storage;
  desc.mappedAtCreation = true;
  WGPUBuffer buf = wgpuDeviceCreateBuffer(g_device, &desc);
  if (buf == nullptr) {
    return Error::make(ErrorCode::Internal, "WebGPU buffer create failed");
  }
  void* mapped = wgpuBufferGetMappedRange(buf, 0, len);
  if (mapped == nullptr) {
    wgpuBufferDestroy(buf);
    wgpuBufferRelease(buf);
    return Error::make(ErrorCode::Internal, "WebGPU buffer map failed");
  }
  BufferRec rec{buf, mapped, len};
  g_bufs[mapped] = rec;
  *out_ptr = mapped;
  return Error::success();
#endif
}

Error webgpu_free(void* ptr) noexcept {
#if !defined(UAII_HAVE_WEBGPU_HEADERS)
  (void)ptr;
  return Error::success();
#else
  if (ptr == nullptr) {
    return Error::success();
  }
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_bufs.find(ptr);
  if (it == g_bufs.end()) {
    return Error::make(ErrorCode::NotFound, "WebGPU pointer not owned");
  }
  if (it->second.buffer != nullptr) {
    wgpuBufferUnmap(it->second.buffer);
    wgpuBufferDestroy(it->second.buffer);
    wgpuBufferRelease(it->second.buffer);
  }
  g_bufs.erase(it);
  return Error::success();
#endif
}

Error webgpu_copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (bytes == 0) return Error::success();
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "webgpu copy_h2d null");
  }
  std::memcpy(device, host, bytes);
  return Error::success();
}

Error webgpu_copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (bytes == 0) return Error::success();
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "webgpu copy_d2h null");
  }
  std::memcpy(host, device, bytes);
  return Error::success();
}

Error webgpu_copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (bytes == 0) return Error::success();
  if (src == nullptr || dst == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "webgpu copy_d2d null");
  }
  std::memcpy(dst, src, bytes);
  return Error::success();
}

Error webgpu_synchronize() {
#if defined(UAII_HAVE_WEBGPU_HEADERS)
  if (g_queue != nullptr) {
    wgpuQueueSubmit(g_queue, 0, nullptr);
  }
#endif
  return Error::success();
}

Error webgpu_dispatch(const std::string&, const std::string&,
                      const std::vector<kernels::TensorView>&,
                      std::vector<kernels::TensorView>*,
                      const std::vector<ir::Attribute>&) {
  return Error::make(ErrorCode::NotImplemented,
                     "WebGPU on-device op kernels not implemented (use host_fallback)");
}

}  // namespace native
}  // namespace backends
}  // namespace uaii
