#include "native_stubs.hpp"
#include "rocm_kernels.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#if defined(__has_include)
#if __has_include(<hip/hip_runtime.h>)
#define UAII_HAVE_HIP_HEADERS 1
#include <hip/hip_runtime.h>
#if __has_include(<rocblas/rocblas.h>)
#define UAII_HAVE_ROCBLAS 1
#include <rocblas/rocblas.h>
#endif
#endif
#endif

namespace uaii {
namespace backends {
namespace native {
namespace {

#if defined(UAII_HAVE_HIP_HEADERS)
std::mutex g_mu;
bool g_inited = false;
hipStream_t g_stream = nullptr;
#if defined(UAII_HAVE_ROCBLAS)
rocblas_handle g_blas = nullptr;
#endif
std::unordered_map<void*, std::size_t> g_allocs;

Error hip_status(hipError_t st, const char* what) {
  if (st == hipSuccess) {
    return Error::success();
  }
  return Error::make(ErrorCode::Internal,
                     std::string("HIP ") + what + ": " + hipGetErrorString(st));
}
#endif

bool attr_bool(const std::vector<ir::Attribute>& attrs, const char* key, bool def) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::Bool) {
      return std::get<bool>(a.value);
    }
  }
  return def;
}

float attr_float(const std::vector<ir::Attribute>& attrs, const char* key, float def) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::Float) {
      return static_cast<float>(std::get<double>(a.value));
    }
    if (a.key == key && a.type == ir::AttributeType::Int) {
      return static_cast<float>(std::get<std::int64_t>(a.value));
    }
  }
  return def;
}

#if defined(UAII_HAVE_HIP_HEADERS)
Error require_f32(const kernels::TensorView& t, const char* name) {
  if (t.dtype != DType::F32 || t.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string("ROCm ") + name + " requires f32 data");
  }
  return Error::success();
}

Error dispatch_add(const std::vector<kernels::TensorView>& inputs,
                   std::vector<kernels::TensorView>* outputs) {
  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm Add arity");
  }
  Error err = require_f32(inputs[0], "A");
  if (!err.ok()) return err;
  err = require_f32(inputs[1], "B");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "C");
  if (!err.ok()) return err;
  if (inputs[0].numel() != inputs[1].numel() ||
      inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm Add size");
  }
  const int st = rocm_kernels::launch_add_f32(inputs[0].f32(), inputs[1].f32(),
                                              (*outputs)[0].f32(), inputs[0].numel(), g_stream);
  return hip_status(static_cast<hipError_t>(st), "Add");
}

Error dispatch_rmsnorm(const std::vector<kernels::TensorView>& inputs,
                       std::vector<kernels::TensorView>* outputs,
                       const std::vector<ir::Attribute>& attrs) {
  if (inputs.empty() || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm RMSNorm arity");
  }
  Error err = require_f32(inputs[0], "x");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "y");
  if (!err.ok()) return err;
  if (inputs[0].rank < 1) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm RMSNorm rank");
  }
  const std::int64_t cols = inputs[0].dim(inputs[0].rank - 1);
  std::int64_t rows = 1;
  for (std::size_t i = 0; i + 1 < inputs[0].rank; ++i) {
    rows *= inputs[0].dim(i);
  }
  if (rows * cols != static_cast<std::int64_t>(inputs[0].numel()) ||
      inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm RMSNorm size");
  }
  const float* w = nullptr;
  if (inputs.size() > 1 && inputs[1].data != nullptr) {
    err = require_f32(inputs[1], "weight");
    if (!err.ok()) return err;
    if (static_cast<std::int64_t>(inputs[1].numel()) != cols) {
      return Error::make(ErrorCode::InvalidArgument, "ROCm RMSNorm weight");
    }
    w = inputs[1].f32();
  }
  const float eps = attr_float(attrs, "eps", 1e-5f);
  const int st = rocm_kernels::launch_rmsnorm_f32(inputs[0].f32(), w, (*outputs)[0].f32(), rows,
                                                  cols, eps, g_stream);
  return hip_status(static_cast<hipError_t>(st), "RMSNorm");
}
#endif

}  // namespace

bool rocm_compiled() noexcept { return true; }

const char* rocm_capability_details() noexcept {
#if defined(UAII_HAVE_HIP_HEADERS) && defined(UAII_HAVE_ROCBLAS)
  return "ROCm native: MatMul(rocBLAS),Add,RMSNorm(HIP); host_fallback: "
         "Softmax,Silu,Attention,others";
#elif defined(UAII_HAVE_HIP_HEADERS)
  return "ROCm HIP buffers; host_fallback: MatMul,Add,RMSNorm,Softmax,Silu,Attention,"
         "others (rocBLAS not found)";
#else
  return "ROCm/HIP headers unavailable — host_fallback: all ops";
#endif
}

Error rocm_init() {
#if !defined(UAII_HAVE_HIP_HEADERS)
  return Error::make(ErrorCode::NotImplemented,
                     "HIP headers not found (install ROCm / hip/hip_runtime.h)");
#else
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_inited) {
    return Error::success();
  }
  int count = 0;
  Error err = hip_status(hipGetDeviceCount(&count), "GetDeviceCount");
  if (!err.ok()) {
    return err;
  }
  if (count <= 0) {
    return Error::make(ErrorCode::NotFound, "no HIP devices");
  }
  err = hip_status(hipSetDevice(0), "SetDevice");
  if (!err.ok()) {
    return err;
  }
  err = hip_status(hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking),
                   "StreamCreate");
  if (!err.ok()) {
    return err;
  }
#if defined(UAII_HAVE_ROCBLAS)
  if (rocblas_create_handle(&g_blas) != rocblas_status_success) {
    hipStreamDestroy(g_stream);
    g_stream = nullptr;
    return Error::make(ErrorCode::Internal, "rocblas_create_handle failed");
  }
  rocblas_set_stream(g_blas, g_stream);
#endif
  g_inited = true;
  return Error::success();
#endif
}

void rocm_shutdown() noexcept {
#if defined(UAII_HAVE_HIP_HEADERS)
  std::lock_guard<std::mutex> lock(g_mu);
  for (auto& kv : g_allocs) {
    hipFree(kv.first);
  }
  g_allocs.clear();
#if defined(UAII_HAVE_ROCBLAS)
  if (g_blas) {
    rocblas_destroy_handle(g_blas);
    g_blas = nullptr;
  }
#endif
  if (g_stream) {
    hipStreamDestroy(g_stream);
    g_stream = nullptr;
  }
  g_inited = false;
#endif
}

Error rocm_allocate(std::size_t bytes, void** out_ptr) {
#if !defined(UAII_HAVE_HIP_HEADERS)
  (void)bytes;
  (void)out_ptr;
  return rocm_init();
#else
  if (out_ptr == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "rocm_allocate out null");
  }
  if (!g_inited) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm not initialized");
  }
  void* p = nullptr;
  Error err = hip_status(hipMalloc(&p, bytes == 0 ? 1 : bytes), "Malloc");
  if (!err.ok()) {
    return err;
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_allocs[p] = bytes;
  }
  *out_ptr = p;
  return Error::success();
#endif
}

Error rocm_free(void* ptr) noexcept {
#if !defined(UAII_HAVE_HIP_HEADERS)
  (void)ptr;
  return Error::success();
#else
  if (ptr == nullptr) {
    return Error::success();
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_allocs.erase(ptr);
  }
  return hip_status(hipFree(ptr), "Free");
#endif
}

Error rocm_copy_h2d(const void* host, void* device, std::size_t bytes) {
#if !defined(UAII_HAVE_HIP_HEADERS)
  (void)host;
  (void)device;
  (void)bytes;
  return rocm_init();
#else
  if (bytes == 0) return Error::success();
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "rocm copy_h2d null");
  }
  return hip_status(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice), "H2D");
#endif
}

Error rocm_copy_d2h(const void* device, void* host, std::size_t bytes) {
#if !defined(UAII_HAVE_HIP_HEADERS)
  (void)device;
  (void)host;
  (void)bytes;
  return rocm_init();
#else
  if (bytes == 0) return Error::success();
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "rocm copy_d2h null");
  }
  return hip_status(hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost), "D2H");
#endif
}

Error rocm_copy_d2d(const void* src, void* dst, std::size_t bytes) {
#if !defined(UAII_HAVE_HIP_HEADERS)
  (void)src;
  (void)dst;
  (void)bytes;
  return rocm_init();
#else
  if (bytes == 0) return Error::success();
  if (src == nullptr || dst == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "rocm copy_d2d null");
  }
  return hip_status(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice), "D2D");
#endif
}

Error rocm_synchronize() {
#if !defined(UAII_HAVE_HIP_HEADERS)
  return Error::success();
#else
  if (!g_inited) {
    return Error::success();
  }
  return hip_status(hipStreamSynchronize(g_stream), "StreamSynchronize");
#endif
}

Error rocm_dispatch(const std::string& op_name, const std::string&,
                    const std::vector<kernels::TensorView>& inputs,
                    std::vector<kernels::TensorView>* outputs,
                    const std::vector<ir::Attribute>& attrs) {
#if !defined(UAII_HAVE_HIP_HEADERS)
  (void)op_name;
  (void)inputs;
  (void)outputs;
  (void)attrs;
  return Error::make(ErrorCode::NotImplemented,
                     "ROCm on-device op not available (use host_fallback)");
#else
  if (!g_inited) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm not initialized");
  }
  if (op_name == "Add") {
    return dispatch_add(inputs, outputs);
  }
  if (op_name == "RMSNorm") {
    return dispatch_rmsnorm(inputs, outputs, attrs);
  }
  if (op_name != "MatMul") {
    return Error::make(ErrorCode::NotImplemented,
                       "ROCm native op not implemented: " + op_name);
  }
#if !defined(UAII_HAVE_ROCBLAS)
  return Error::make(ErrorCode::NotImplemented, "ROCm MatMul requires rocBLAS");
#else
  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm MatMul arity");
  }
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  auto& c = (*outputs)[0];
  if (a.rank != 2 || b.rank != 2 || c.rank != 2 || a.dtype != DType::F32 ||
      b.dtype != DType::F32 || c.dtype != DType::F32) {
    return Error::make(ErrorCode::InvalidArgument, "ROCm MatMul expects rank-2 f32");
  }
  const bool ta = attr_bool(attrs, "transpose_a", false);
  const bool tb = attr_bool(attrs, "transpose_b", false);
  const std::int64_t m = ta ? a.dim(1) : a.dim(0);
  const std::int64_t k = ta ? a.dim(0) : a.dim(1);
  const std::int64_t n = tb ? b.dim(0) : b.dim(1);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const rocblas_operation op_b = tb ? rocblas_operation_transpose : rocblas_operation_none;
  const rocblas_operation op_a = ta ? rocblas_operation_transpose : rocblas_operation_none;
  const rocblas_status st =
      rocblas_sgemm(g_blas, op_b, op_a, static_cast<int>(n), static_cast<int>(m),
                    static_cast<int>(k), &alpha, b.f32(), static_cast<int>(b.dim(1)), a.f32(),
                    static_cast<int>(a.dim(1)), &beta, c.f32(), static_cast<int>(c.dim(1)));
  if (st != rocblas_status_success) {
    return Error::make(ErrorCode::Internal, "rocblas_sgemm failed");
  }
  return Error::success();
#endif
#endif
}

}  // namespace native
}  // namespace backends
}  // namespace uaii
