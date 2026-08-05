#include "native_stubs.hpp"

#include "cuda_kernels.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace uaii {
namespace backends {
namespace native {
namespace {

std::mutex g_mu;
bool g_inited = false;
cudaStream_t g_stream = nullptr;
cudaStream_t g_copy_stream = nullptr;
cublasHandle_t g_cublas = nullptr;
cublasLtHandle_t g_cublaslt = nullptr;
std::unordered_map<void*, std::size_t> g_device_allocs;

Error cuda_status(cudaError_t st, const char* what) {
  if (st == cudaSuccess) {
    return Error::success();
  }
  return Error::make(ErrorCode::Internal,
                     std::string("CUDA ") + what + ": " + cudaGetErrorString(st));
}

Error cublas_status(cublasStatus_t st, const char* what) {
  if (st == CUBLAS_STATUS_SUCCESS) {
    return Error::success();
  }
  return Error::make(ErrorCode::Internal, std::string("cuBLAS ") + what + " failed");
}

Error cublaslt_status(cublasStatus_t st, const char* what) {
  if (st == CUBLAS_STATUS_SUCCESS) {
    return Error::success();
  }
  return Error::make(ErrorCode::Internal, std::string("cuBLASLt ") + what + " failed");
}

bool attr_bool(const std::vector<ir::Attribute>& attrs, const char* key, bool def) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::Bool) {
      return std::get<bool>(a.value);
    }
  }
  return def;
}

std::int64_t attr_int(const std::vector<ir::Attribute>& attrs, const char* key,
                      std::int64_t def) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::Int) {
      return std::get<std::int64_t>(a.value);
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

Error require_f32(const kernels::TensorView& t, const char* name) {
  if (t.dtype != DType::F32 || t.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument,
                       std::string("CUDA ") + name + " requires f32 data");
  }
  return Error::success();
}

// Row-major GEMM via cuBLASLt (primary).
Error gemm_rowmajor_f32_lt(std::int64_t m, std::int64_t n, std::int64_t k, const float* a,
                           std::int64_t lda, bool transpose_a, const float* b, std::int64_t ldb,
                           bool transpose_b, float* c, std::int64_t ldc) {
  if (g_cublaslt == nullptr) {
    return Error::make(ErrorCode::Internal, "cuBLASLt handle unavailable");
  }

  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;

  const cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int mi = static_cast<int>(m);
  const int ni = static_cast<int>(n);
  const int ki = static_cast<int>(k);
  const int ldai = static_cast<int>(lda);
  const int ldbi = static_cast<int>(ldb);
  const int ldci = static_cast<int>(ldc);
  const int a_rows = (op_a == CUBLAS_OP_N) ? mi : ki;
  const int a_cols = (op_a == CUBLAS_OP_N) ? ki : mi;
  const int b_rows = (op_b == CUBLAS_OP_N) ? ki : ni;
  const int b_cols = (op_b == CUBLAS_OP_N) ? ni : ki;

  auto cleanup = [&]() {
    if (c_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (matmul_desc != nullptr) {
      cublasLtMatmulDescDestroy(matmul_desc);
    }
  };

  cublasStatus_t st =
      cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (st != CUBLAS_STATUS_SUCCESS) {
    return cublaslt_status(st, "MatmulDescCreate");
  }
  st = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                      sizeof(op_a));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "SetAttribute TRANSA");
  }
  st = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                      sizeof(op_b));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "SetAttribute TRANSB");
  }

  st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, a_rows, a_cols, ldai);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "MatrixLayoutCreate A");
  }
  st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, b_rows, b_cols, ldbi);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "MatrixLayoutCreate B");
  }
  st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, mi, ni, ldci);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "MatrixLayoutCreate C");
  }

  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                        sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "SetAttribute ORDER A");
  }
  st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                        sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "SetAttribute ORDER B");
  }
  st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                        sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status(st, "SetAttribute ORDER C");
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  st = cublasLtMatmul(g_cublaslt, matmul_desc, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc,
                      c, c_desc, nullptr, nullptr, 0, g_stream);
  cleanup();
  if (st != CUBLAS_STATUS_SUCCESS) {
    return cublaslt_status(st, "Matmul");
  }
  return Error::success();
}

// Row-major GEMM via cuBLAS (column-major API): C = op(A) @ op(B).
Error gemm_rowmajor_f32_sgemm(std::int64_t m, std::int64_t n, std::int64_t k, const float* a,
                              std::int64_t lda, bool transpose_a, const float* b, std::int64_t ldb,
                              bool transpose_b, float* c, std::int64_t ldc) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Compute C^T = op(B)^T @ op(A)^T in column-major terms.
  const cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  return cublas_status(
      cublasSgemm(g_cublas, op_b, op_a, static_cast<int>(n), static_cast<int>(m),
                  static_cast<int>(k), &alpha, b, static_cast<int>(ldb), a,
                  static_cast<int>(lda), &beta, c, static_cast<int>(ldc)),
      "Sgemm");
}

Error gemm_rowmajor_f32(std::int64_t m, std::int64_t n, std::int64_t k, const float* a,
                        std::int64_t lda, bool transpose_a, const float* b, std::int64_t ldb,
                        bool transpose_b, float* c, std::int64_t ldc) {
  Error err = gemm_rowmajor_f32_lt(m, n, k, a, lda, transpose_a, b, ldb, transpose_b, c, ldc);
  if (err.ok()) {
    return err;
  }
  return gemm_rowmajor_f32_sgemm(m, n, k, a, lda, transpose_a, b, ldb, transpose_b, c, ldc);
}

Error dispatch_matmul(const std::vector<kernels::TensorView>& inputs,
                      std::vector<kernels::TensorView>* outputs,
                      const std::vector<ir::Attribute>& attrs) {
  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA MatMul arity");
  }
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  auto& c = (*outputs)[0];
  Error err = require_f32(a, "A");
  if (!err.ok()) return err;
  err = require_f32(b, "B");
  if (!err.ok()) return err;
  err = require_f32(c, "C");
  if (!err.ok()) return err;
  if (a.rank != 2 || b.rank != 2 || c.rank != 2) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA MatMul expects rank-2");
  }

  const bool ta = attr_bool(attrs, "transpose_a", false);
  const bool tb = attr_bool(attrs, "transpose_b", false);
  const std::int64_t a_rows = ta ? a.dim(1) : a.dim(0);
  const std::int64_t a_cols = ta ? a.dim(0) : a.dim(1);
  const std::int64_t b_rows = tb ? b.dim(1) : b.dim(0);
  const std::int64_t b_cols = tb ? b.dim(0) : b.dim(1);
  if (a_cols != b_rows || c.dim(0) != a_rows || c.dim(1) != b_cols) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA MatMul shape mismatch");
  }
  return gemm_rowmajor_f32(a_rows, b_cols, a_cols, a.f32(), a.dim(1), ta, b.f32(), b.dim(1),
                           tb, c.f32(), c.dim(1));
}

Error dispatch_binary(const char* op, const std::vector<kernels::TensorView>& inputs,
                      std::vector<kernels::TensorView>* outputs) {
  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, std::string("CUDA ") + op + " arity");
  }
  Error err = require_f32(inputs[0], "a");
  if (!err.ok()) return err;
  err = require_f32(inputs[1], "b");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "c");
  if (!err.ok()) return err;
  if (inputs[0].numel() != inputs[1].numel() ||
      inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, std::string("CUDA ") + op + " size");
  }
  const int st =
      (std::strcmp(op, "Add") == 0)
          ? cuda_kernels::launch_add_f32(inputs[0].f32(), inputs[1].f32(),
                                         (*outputs)[0].f32(), inputs[0].numel(), g_stream)
          : cuda_kernels::launch_mul_f32(inputs[0].f32(), inputs[1].f32(),
                                         (*outputs)[0].f32(), inputs[0].numel(), g_stream);
  return cuda_status(static_cast<cudaError_t>(st), op);
}

Error dispatch_silu(const std::vector<kernels::TensorView>& inputs,
                    std::vector<kernels::TensorView>* outputs) {
  if (inputs.size() != 1 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Silu arity");
  }
  Error err = require_f32(inputs[0], "x");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "y");
  if (!err.ok()) return err;
  if (inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Silu size");
  }
  const int st = cuda_kernels::launch_silu_f32(inputs[0].f32(), (*outputs)[0].f32(),
                                               inputs[0].numel(), g_stream);
  return cuda_status(static_cast<cudaError_t>(st), "Silu");
}

Error dispatch_rmsnorm(const std::vector<kernels::TensorView>& inputs,
                       std::vector<kernels::TensorView>* outputs,
                       const std::vector<ir::Attribute>& attrs) {
  if (inputs.empty() || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RMSNorm arity");
  }
  Error err = require_f32(inputs[0], "x");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "y");
  if (!err.ok()) return err;
  if (inputs[0].rank < 1) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RMSNorm rank");
  }
  const std::int64_t cols = inputs[0].dim(inputs[0].rank - 1);
  std::int64_t rows = 1;
  for (std::size_t i = 0; i + 1 < inputs[0].rank; ++i) {
    rows *= inputs[0].dim(i);
  }
  if (rows * cols != static_cast<std::int64_t>(inputs[0].numel()) ||
      inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RMSNorm size");
  }
  const float* w = nullptr;
  if (inputs.size() > 1) {
    err = require_f32(inputs[1], "weight");
    if (!err.ok()) return err;
    if (static_cast<std::int64_t>(inputs[1].numel()) != cols) {
      return Error::make(ErrorCode::InvalidArgument, "CUDA RMSNorm weight");
    }
    w = inputs[1].f32();
  }
  const float eps = attr_float(attrs, "eps", 1e-5f);
  const int st = cuda_kernels::launch_rmsnorm_f32(inputs[0].f32(), w, (*outputs)[0].f32(),
                                                  rows, cols, eps, g_stream);
  return cuda_status(static_cast<cudaError_t>(st), "RMSNorm");
}

Error dispatch_softmax(const std::vector<kernels::TensorView>& inputs,
                       std::vector<kernels::TensorView>* outputs,
                       const std::vector<ir::Attribute>& attrs) {
  if (inputs.size() != 1 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Softmax arity");
  }
  Error err = require_f32(inputs[0], "x");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "y");
  if (!err.ok()) return err;
  if (inputs[0].numel() != (*outputs)[0].numel() || inputs[0].rank == 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Softmax size");
  }
  int axis = static_cast<int>(attr_int(attrs, "axis", -1));
  if (axis < 0) {
    axis += static_cast<int>(inputs[0].rank);
  }
  if (axis < 0 || axis >= static_cast<int>(inputs[0].rank)) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Softmax axis");
  }
  std::int64_t outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= inputs[0].dim(static_cast<std::size_t>(i));
  }
  const std::int64_t ax = inputs[0].dim(static_cast<std::size_t>(axis));
  std::int64_t inner = 1;
  for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < inputs[0].rank; ++i) {
    inner *= inputs[0].dim(i);
  }
  const int st = cuda_kernels::launch_softmax_f32(inputs[0].f32(), (*outputs)[0].f32(),
                                                  outer, ax, inner, g_stream);
  return cuda_status(static_cast<cudaError_t>(st), "Softmax");
}

}  // namespace

bool cuda_compiled() noexcept { return true; }

const char* cuda_capability_details() noexcept {
  return "CUDA native: MatMul(cuBLASLt+cublasSgemm fallback),Add,Mul,RMSNorm,Softmax,Silu "
         "on-device; cuBLASLt + dual-stream; host_fallback: Attention,RoPE,Embedding,others";
}

Error cuda_init() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_inited) {
    return Error::success();
  }
  int count = 0;
  Error err = cuda_status(cudaGetDeviceCount(&count), "GetDeviceCount");
  if (!err.ok()) {
    return err;
  }
  if (count <= 0) {
    return Error::make(ErrorCode::NotFound, "no CUDA devices");
  }
  err = cuda_status(cudaSetDevice(0), "SetDevice");
  if (!err.ok()) {
    return err;
  }
  err = cuda_status(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking),
                    "StreamCreate");
  if (!err.ok()) {
    return err;
  }
  err = cuda_status(cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking),
                    "CopyStreamCreate");
  if (!err.ok()) {
    cudaStreamDestroy(g_stream);
    g_stream = nullptr;
    return err;
  }
  err = cublas_status(cublasCreate(&g_cublas), "Create");
  if (!err.ok()) {
    cudaStreamDestroy(g_copy_stream);
    g_copy_stream = nullptr;
    cudaStreamDestroy(g_stream);
    g_stream = nullptr;
    return err;
  }
  err = cublas_status(cublasSetStream(g_cublas, g_stream), "SetStream");
  if (!err.ok()) {
    cublasDestroy(g_cublas);
    g_cublas = nullptr;
    cudaStreamDestroy(g_copy_stream);
    g_copy_stream = nullptr;
    cudaStreamDestroy(g_stream);
    g_stream = nullptr;
    return err;
  }
  if (cublasLtCreate(&g_cublaslt) != CUBLAS_STATUS_SUCCESS) {
    g_cublaslt = nullptr;
  }
  g_inited = true;
  return Error::success();
}

void cuda_shutdown() noexcept {
  std::lock_guard<std::mutex> lock(g_mu);
  for (auto& kv : g_device_allocs) {
    cudaFree(kv.first);
  }
  g_device_allocs.clear();
  if (g_cublas != nullptr) {
    cublasDestroy(g_cublas);
    g_cublas = nullptr;
  }
  if (g_cublaslt != nullptr) {
    cublasLtDestroy(g_cublaslt);
    g_cublaslt = nullptr;
  }
  if (g_copy_stream != nullptr) {
    cudaStreamDestroy(g_copy_stream);
    g_copy_stream = nullptr;
  }
  if (g_stream != nullptr) {
    cudaStreamDestroy(g_stream);
    g_stream = nullptr;
  }
  g_inited = false;
}

Error cuda_allocate(std::size_t bytes, void** out_ptr) {
  if (out_ptr == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "cuda_allocate out null");
  }
  if (!g_inited) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA not initialized");
  }
  void* p = nullptr;
  Error err = cuda_status(cudaMalloc(&p, bytes == 0 ? 1 : bytes), "Malloc");
  if (!err.ok()) {
    return err;
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_device_allocs[p] = bytes;
  }
  *out_ptr = p;
  return Error::success();
}

Error cuda_free(void* ptr) noexcept {
  if (ptr == nullptr) {
    return Error::success();
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_device_allocs.erase(ptr);
  }
  return cuda_status(cudaFree(ptr), "Free");
}

Error cuda_copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (bytes == 0) {
    return Error::success();
  }
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "cuda copy_h2d null");
  }
  // Synchronous copies so host-staging paths can read/write immediately.
  return cuda_status(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "H2D");
}

Error cuda_copy_h2d_async(const void* host, void* device, std::size_t bytes) {
  if (bytes == 0) {
    return Error::success();
  }
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "cuda copy_h2d_async null");
  }
  if (!g_inited) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA not initialized");
  }
  return cuda_status(
      cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, g_copy_stream), "H2D async");
}

Error cuda_copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (bytes == 0) {
    return Error::success();
  }
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "cuda copy_d2h null");
  }
  return cuda_status(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "D2H");
}

Error cuda_copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (bytes == 0) {
    return Error::success();
  }
  if (src == nullptr || dst == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "cuda copy_d2d null");
  }
  return cuda_status(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "D2D");
}

Error cuda_synchronize() {
  if (!g_inited) {
    return Error::success();
  }
  Error err = cuda_status(cudaStreamSynchronize(g_stream), "StreamSynchronize");
  if (!err.ok()) {
    return err;
  }
  return cuda_status(cudaStreamSynchronize(g_copy_stream), "CopyStreamSynchronize");
}

Error cuda_dispatch(const std::string& op_name, const std::string& /*op_version*/,
                    const std::vector<kernels::TensorView>& inputs,
                    std::vector<kernels::TensorView>* outputs,
                    const std::vector<ir::Attribute>& attrs) {
  if (!g_inited) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA not initialized");
  }
  if (op_name == "MatMul" || op_name == "MatMulRelu") {
    Error err = dispatch_matmul(inputs, outputs, attrs);
    if (!err.ok()) {
      return err;
    }
    if (op_name == "MatMulRelu") {
      auto& c = (*outputs)[0];
      const int st =
          cuda_kernels::launch_relu_f32(c.f32(), c.f32(), c.numel(), g_stream);
      return cuda_status(static_cast<cudaError_t>(st), "MatMulRelu");
    }
    return Error::success();
  }
  if (op_name == "Add") {
    return dispatch_binary("Add", inputs, outputs);
  }
  if (op_name == "Mul") {
    return dispatch_binary("Mul", inputs, outputs);
  }
  if (op_name == "RMSNorm") {
    return dispatch_rmsnorm(inputs, outputs, attrs);
  }
  if (op_name == "Softmax") {
    return dispatch_softmax(inputs, outputs, attrs);
  }
  if (op_name == "Silu") {
    return dispatch_silu(inputs, outputs);
  }
  return Error::make(ErrorCode::NotImplemented,
                     "CUDA native op not implemented: " + op_name);
}

bool cuda_probe_device() noexcept {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace native
}  // namespace backends
}  // namespace uaii
