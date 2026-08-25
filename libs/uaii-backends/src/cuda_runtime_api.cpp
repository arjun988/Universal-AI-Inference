#include "native_stubs.hpp"

#include "cuda_kernels.hpp"

#include "uaii/kernels/kernels.hpp"
#include "uaii/runtime/kv_cache.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

std::string attr_string(const std::vector<ir::Attribute>& attrs, const char* key) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::String) {
      return std::get<std::string>(a.value);
    }
  }
  return {};
}

std::vector<float> attr_float_array(const std::vector<ir::Attribute>& attrs, const char* key) {
  for (const auto& a : attrs) {
    if (a.key == key && a.type == ir::AttributeType::FloatArray) {
      const auto& src = std::get<std::vector<double>>(a.value);
      std::vector<float> out(src.size());
      for (std::size_t i = 0; i < src.size(); ++i) out[i] = static_cast<float>(src[i]);
      return out;
    }
  }
  return {};
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

Error gemm_rowmajor_f32_alpha(std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              const float* a, std::int64_t lda, bool transpose_a, const float* b,
                              std::int64_t ldb, bool transpose_b, float* c, std::int64_t ldc) {
  // Prefer sgemm with alpha for attention scores (lt path keeps alpha=1).
  const float beta = 0.0f;
  const cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  return cublas_status(
      cublasSgemm(g_cublas, op_b, op_a, static_cast<int>(n), static_cast<int>(m),
                  static_cast<int>(k), &alpha, b, static_cast<int>(ldb), a,
                  static_cast<int>(lda), &beta, c, static_cast<int>(ldc)),
      "Sgemm alpha");
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
  const float offset = attr_float(attrs, "offset", 0.f);
  if (offset != 0.f || (inputs.size() > 1 &&
                        static_cast<std::int64_t>(inputs[1].numel()) != cols)) {
    return Error::make(ErrorCode::NotImplemented,
                       "CUDA RMSNorm offset/QK-norm uses CPU kernel");
  }
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

struct DeviceBuf {
  float* p = nullptr;
  std::size_t bytes = 0;
  bool owned = false;

  void release() noexcept {
    if (owned && p != nullptr) {
      cudaFree(p);
    }
    p = nullptr;
    bytes = 0;
    owned = false;
  }
};

Error device_buf_alloc(DeviceBuf* b, std::size_t bytes) {
  if (b == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "device_buf null");
  }
  b->release();
  if (bytes == 0) {
    return Error::success();
  }
  void* p = nullptr;
  Error err = cuda_status(cudaMalloc(&p, bytes), "Malloc temp");
  if (!err.ok()) {
    return err;
  }
  b->p = static_cast<float*>(p);
  b->bytes = bytes;
  b->owned = true;
  return Error::success();
}

// Ensure `t` is addressable on device: reuse if already device, else alloc+H2D.
Error ensure_device_f32(const kernels::TensorView& t, DeviceBuf* scratch, const float** out) {
  if (out == nullptr || scratch == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "ensure_device null");
  }
  Error err = require_f32(t, "tensor");
  if (!err.ok()) return err;
  if (cuda_is_device_ptr(t.data)) {
    *out = t.f32();
    return Error::success();
  }
  err = device_buf_alloc(scratch, t.nbytes);
  if (!err.ok()) return err;
  err = cuda_status(cudaMemcpy(scratch->p, t.data, t.nbytes, cudaMemcpyHostToDevice), "H2D");
  if (!err.ok()) return err;
  *out = scratch->p;
  return Error::success();
}

Error ensure_device_out_f32(kernels::TensorView* t, DeviceBuf* scratch, float** out,
                           bool* need_d2h) {
  if (out == nullptr || scratch == nullptr || need_d2h == nullptr || t == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "ensure_device_out null");
  }
  Error err = require_f32(*t, "out");
  if (!err.ok()) return err;
  if (cuda_is_device_ptr(t->data)) {
    *out = t->f32();
    *need_d2h = false;
    return Error::success();
  }
  err = device_buf_alloc(scratch, t->nbytes);
  if (!err.ok()) return err;
  *out = scratch->p;
  *need_d2h = true;
  return Error::success();
}

struct AttnShape {
  std::int64_t batch = 0;
  std::int64_t seq = 0;
  std::int64_t dim = 0;
};

Error parse_attn_shape(const kernels::TensorView& t, AttnShape* out, const char* name) {
  if (out == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "attn shape null");
  }
  if (t.dtype != DType::F32 || t.data == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, std::string(name) + " f32 required");
  }
  if (t.rank == 2) {
    out->batch = t.dim(0);
    out->seq = 1;
    out->dim = t.dim(1);
    return Error::success();
  }
  if (t.rank == 3) {
    out->batch = t.dim(0);
    out->seq = t.dim(1);
    out->dim = t.dim(2);
    return Error::success();
  }
  return Error::make(ErrorCode::InvalidArgument,
                     std::string(name) + " expects [batch,seq,dim] or [batch,dim]");
}

Error dispatch_rope(const std::vector<kernels::TensorView>& inputs,
                    std::vector<kernels::TensorView>* outputs,
                    const std::vector<ir::Attribute>& attrs) {
  if (inputs.empty() || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RoPE arity");
  }
  const std::string scale = attr_string(attrs, "scale_type");
  const auto& in = inputs[0];
  auto& out = (*outputs)[0];
  Error err = require_f32(in, "rope_in");
  if (!err.ok()) return err;
  err = require_f32(out, "rope_out");
  if (!err.ok()) return err;
  if (in.numel() != out.numel() || in.rank < 2) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RoPE shape");
  }
  const std::int64_t dim = in.dim(in.rank - 1);
  if (dim % 2 != 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RoPE dim must be even");
  }
  const std::int64_t head_dim = attr_int(attrs, "head_dim", 0);
  std::int64_t rope_chunk = head_dim > 0 ? head_dim : dim;
  if (rope_chunk <= 0 || dim % rope_chunk != 0 || rope_chunk % 2 != 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RoPE head_dim invalid");
  }
  const std::int64_t rotary_dim_attr = attr_int(attrs, "rotary_dim", 0);
  std::int64_t rot = rotary_dim_attr > 0 ? rotary_dim_attr : rope_chunk;
  if (rot > rope_chunk || rot <= 0 || rot % 2 != 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA RoPE rotary_dim invalid");
  }
  const float theta = attr_float(attrs, "theta", 10000.f);
  const int mode = static_cast<int>(attr_int(attrs, "mode", 0));
  float base_pos = attr_float(attrs, "position", -1.f);
  if (base_pos < 0.f) {
    base_pos = attr_float(attrs, "past_len", -1.f);
  }
  if (base_pos < 0.f) {
    runtime::KvCache* cache = kernels::active_kv_cache();
    base_pos = (cache != nullptr) ? static_cast<float>(cache->rope_pos()) : 0.f;
  }
  const std::int64_t rows =
      static_cast<std::int64_t>(in.numel() / static_cast<std::size_t>(dim));

  DeviceBuf d_in, d_out, d_pos;
  const float* x = nullptr;
  float* y = nullptr;
  bool need_d2h = false;
  err = ensure_device_f32(in, &d_in, &x);
  if (!err.ok()) {
    d_in.release();
    return err;
  }
  err = ensure_device_out_f32(&out, &d_out, &y, &need_d2h);
  if (!err.ok()) {
    d_in.release();
    d_out.release();
    return err;
  }

  const float* pos_dev = nullptr;
  if (inputs.size() > 1 && inputs[1].data != nullptr && inputs[1].numel() > 0) {
    err = require_f32(inputs[1], "rope_pos");
    if (!err.ok()) {
      d_in.release();
      d_out.release();
      return err;
    }
    err = ensure_device_f32(inputs[1], &d_pos, &pos_dev);
    if (!err.ok()) {
      d_in.release();
      d_out.release();
      d_pos.release();
      return err;
    }
  }

  if (!scale.empty() && scale != "none" && scale != "default" && scale != "linear" &&
      scale != "llama3" && scale != "dynamic" && scale != "yarn" && scale != "longrope" &&
      scale != "su") {
    d_in.release();
    d_out.release();
    d_pos.release();
    return Error::make(ErrorCode::NotImplemented, "unsupported rope_scaling.type=" + scale);
  }

  const std::int64_t half = rot / 2;
  std::vector<float> inv(static_cast<std::size_t>(std::max<std::int64_t>(half, 0)));
  const float seq_len = base_pos + static_cast<float>(rows);
  const float scale_factor = attr_float(attrs, "scale_factor", 1.f);
  const float low_f = attr_float(attrs, "low_freq_factor", 1.f);
  const float high_f = attr_float(attrs, "high_freq_factor", 4.f);
  const float old_ctx = attr_float(attrs, "old_context_len", 8192.f);
  const auto short_f = attr_float_array(attrs, "short_factor");
  const auto long_f = attr_float_array(attrs, "long_factor");
  const int nfac = static_cast<int>(std::max(short_f.size(), long_f.size()));
  const char* stype = scale.empty() ? nullptr : scale.c_str();
  for (std::int64_t i = 0; i < half; ++i) {
    inv[static_cast<std::size_t>(i)] = kernels::rope_scaled_inv_freq(
        i, half, theta, stype, scale_factor, low_f, high_f, old_ctx, seq_len,
        short_f.empty() ? nullptr : short_f.data(),
        long_f.empty() ? nullptr : long_f.data(), nfac);
  }
  float mscale = 1.f;
  if (scale == "yarn") {
    const float factor = scale_factor > 0.f ? scale_factor : 1.f;
    const float af = attr_float(attrs, "attn_factor", 1.f);
    if (factor > 1.f) mscale = 0.1f * af * std::log(factor) + 1.f;
  }

  DeviceBuf d_inv;
  const float* inv_dev = nullptr;
  if (!inv.empty()) {
    err = device_buf_alloc(&d_inv, inv.size() * sizeof(float));
    if (!err.ok()) {
      d_in.release();
      d_out.release();
      d_pos.release();
      return err;
    }
    err = cuda_status(cudaMemcpy(d_inv.p, inv.data(), d_inv.bytes, cudaMemcpyHostToDevice),
                      "RoPE inv_freq H2D");
    if (!err.ok()) {
      d_in.release();
      d_out.release();
      d_pos.release();
      d_inv.release();
      return err;
    }
    inv_dev = d_inv.p;
  }

  const int st = cuda_kernels::launch_rope_f32(x, y, rows, dim, rope_chunk, rot, mode, theta,
                                               base_pos, pos_dev, inv_dev, mscale, g_stream);
  err = cuda_status(static_cast<cudaError_t>(st), "RoPE");
  if (err.ok()) {
    err = cuda_status(cudaStreamSynchronize(g_stream), "RoPE sync");
  }
  if (err.ok() && need_d2h) {
    err = cuda_status(cudaMemcpy(out.data, y, out.nbytes, cudaMemcpyDeviceToHost), "RoPE D2H");
  }
  d_in.release();
  d_out.release();
  d_pos.release();
  d_inv.release();
  return err;
}

// Decode-friendly Attention: q_seq == new_kv_seq == 1 via cuBLAS Q@K^T + softmax + scores@V.
Error cuda_attention_decode_f32(const float* Q, const float* K_full, const float* V_full,
                                float* O, std::int64_t batch, std::int64_t kv_seq,
                                std::int64_t q_dim, std::int64_t k_dim, std::int64_t v_dim,
                                std::int64_t out_dim, int num_heads, int num_kv_heads,
                                std::int64_t head_dim, std::int64_t v_head_dim, float scale,
                                float logit_softcap) {
  const int heads_per_kv = num_heads / num_kv_heads;
  DeviceBuf d_scores;
  Error err = device_buf_alloc(&d_scores, static_cast<std::size_t>(kv_seq) * sizeof(float));
  if (!err.ok()) {
    return err;
  }

  for (std::int64_t b = 0; b < batch; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      const int kv_h = h / heads_per_kv;
      const float* q_row = Q + (b * q_dim) + static_cast<std::int64_t>(h) * head_dim;
      const float* k_ptr =
          K_full + (b * kv_seq * k_dim) + static_cast<std::int64_t>(kv_h) * head_dim;
      const float* v_ptr =
          V_full + (b * kv_seq * v_dim) + static_cast<std::int64_t>(kv_h) * v_head_dim;
      float* o_row = O + (b * out_dim) + static_cast<std::int64_t>(h) * v_head_dim;

      // scores[1, kv_seq] = scale * Q[1, hd] @ K^T[hd, kv_seq]  (K rows stride = k_dim)
      err = gemm_rowmajor_f32_alpha(1, kv_seq, head_dim, scale, q_row, head_dim, false, k_ptr,
                                    k_dim, true, d_scores.p, kv_seq);
      if (!err.ok()) {
        d_scores.release();
        return err;
      }
      if (logit_softcap > 0.f) {
        const int st = cuda_kernels::launch_logit_softcap_f32(
            d_scores.p, d_scores.p, static_cast<std::size_t>(kv_seq), logit_softcap, g_stream);
        err = cuda_status(static_cast<cudaError_t>(st), "softcap");
        if (!err.ok()) {
          d_scores.release();
          return err;
        }
      }
      {
        const int st = cuda_kernels::launch_softmax_f32(d_scores.p, d_scores.p, 1, kv_seq, 1,
                                                       g_stream);
        err = cuda_status(static_cast<cudaError_t>(st), "attn softmax");
        if (!err.ok()) {
          d_scores.release();
          return err;
        }
      }
      // out[1, v_hd] = scores[1, kv_seq] @ V[kv_seq, v_hd] (V rows stride = v_dim)
      err = gemm_rowmajor_f32(1, v_head_dim, kv_seq, d_scores.p, kv_seq, false, v_ptr, v_dim,
                              false, o_row, v_head_dim);
      if (!err.ok()) {
        d_scores.release();
        return err;
      }
    }
  }
  err = cuda_status(cudaStreamSynchronize(g_stream), "attn sync");
  d_scores.release();
  return err;
}

Error dispatch_attention(const std::vector<kernels::TensorView>& inputs,
                         std::vector<kernels::TensorView>* outputs,
                         const std::vector<ir::Attribute>& attrs) {
  if (inputs.size() < 3 || outputs == nullptr || outputs->empty()) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Attention expects Q,K,V");
  }
  const int num_heads = static_cast<int>(attr_int(attrs, "num_heads", 1));
  float scale = attr_float(attrs, "scale", 0.f);
  const bool causal = attr_bool(attrs, "causal", true);
  (void)causal;  // decode seq=1: all keys are valid under causal
  const bool use_kv = attr_bool(attrs, "use_kv_cache", false);
  const std::int64_t layer_id = attr_int(attrs, "layer_id", -1);
  const float logit_softcap = attr_float(attrs, "logit_softcap", 0.f);
  if (attr_int(attrs, "sliding_window", 0) > 0) {
    return Error::make(ErrorCode::NotImplemented,
                       "CUDA Attention sliding_window uses CPU kernel");
  }

  AttnShape qs, ks, vs, os;
  Error err = parse_attn_shape(inputs[0], &qs, "attn_q");
  if (!err.ok()) return err;
  err = parse_attn_shape(inputs[1], &ks, "attn_k");
  if (!err.ok()) return err;
  err = parse_attn_shape(inputs[2], &vs, "attn_v");
  if (!err.ok()) return err;
  err = parse_attn_shape((*outputs)[0], &os, "attn_out");
  if (!err.ok()) return err;

  if (qs.batch != ks.batch || qs.batch != vs.batch || qs.batch != os.batch ||
      qs.seq != os.seq || ks.seq != vs.seq || qs.seq != ks.seq) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Attention shape mismatch");
  }
  // Prefill / multi-token: fall back to CPU (NotImplemented → host staging).
  if (qs.seq != 1 || ks.seq != 1) {
    return Error::make(ErrorCode::NotImplemented,
                       "CUDA Attention decode path requires seq=1 (prefill uses CPU)");
  }

  const std::int64_t batch = qs.batch;
  const std::int64_t q_dim = qs.dim;
  const std::int64_t k_dim = ks.dim;
  const std::int64_t v_dim = vs.dim;
  if (num_heads <= 0 || q_dim % num_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Attention invalid num_heads");
  }
  const std::int64_t head_dim = q_dim / num_heads;
  if (k_dim % head_dim != 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Attention k_dim");
  }
  const int num_kv_heads = static_cast<int>(k_dim / head_dim);
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Attention GQA");
  }
  if (v_dim % num_kv_heads != 0) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Attention v_dim");
  }
  const std::int64_t v_head_dim = v_dim / num_kv_heads;
  const std::int64_t out_dim = static_cast<std::int64_t>(num_heads) * v_head_dim;
  if (os.dim != out_dim) {
    return Error::make(ErrorCode::InvalidArgument, "CUDA Attention out dim");
  }
  if (scale <= 0) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  }

  const kernels::TensorView* past_k = nullptr;
  const kernels::TensorView* past_v = nullptr;
  kernels::TensorView past_k_view;
  kernels::TensorView past_v_view;
  runtime::KvCache* cache = nullptr;

  if (inputs.size() >= 5) {
    past_k = &inputs[3];
    past_v = &inputs[4];
  } else if (use_kv) {
    cache = kernels::active_kv_cache();
    if (cache != nullptr && layer_id >= 0) {
      past_k_view = cache->k_view(layer_id);
      past_v_view = cache->v_view(layer_id);
      past_k = past_k_view.data != nullptr ? &past_k_view : nullptr;
      past_v = past_v_view.data != nullptr ? &past_v_view : nullptr;
    }
  }

  std::int64_t past_seq = 0;
  if (past_k != nullptr || past_v != nullptr) {
    if (past_k == nullptr || past_v == nullptr || past_k->data == nullptr ||
        past_v->data == nullptr) {
      return Error::make(ErrorCode::InvalidArgument, "CUDA Attention past incomplete");
    }
    AttnShape pks, pvs;
    err = parse_attn_shape(*past_k, &pks, "past_k");
    if (!err.ok()) return err;
    err = parse_attn_shape(*past_v, &pvs, "past_v");
    if (!err.ok()) return err;
    if (pks.batch != batch || pvs.batch != batch || pks.dim != k_dim || pvs.dim != v_dim ||
        pks.seq != pvs.seq) {
      return Error::make(ErrorCode::InvalidArgument, "CUDA Attention past shape");
    }
    past_seq = pks.seq;
  }
  const std::int64_t kv_seq = past_seq + 1;

  DeviceBuf d_q, d_k_full, d_v_full, d_out;
  const float* q_dev = nullptr;
  float* out_dev = nullptr;
  bool need_d2h = false;
  err = ensure_device_f32(inputs[0], &d_q, &q_dev);
  if (!err.ok()) {
    d_q.release();
    return err;
  }
  err = ensure_device_out_f32(&(*outputs)[0], &d_out, &out_dev, &need_d2h);
  if (!err.ok()) {
    d_q.release();
    d_out.release();
    return err;
  }

  const std::size_t k_full_bytes =
      static_cast<std::size_t>(batch * kv_seq * k_dim) * sizeof(float);
  const std::size_t v_full_bytes =
      static_cast<std::size_t>(batch * kv_seq * v_dim) * sizeof(float);
  err = device_buf_alloc(&d_k_full, k_full_bytes);
  if (!err.ok()) {
    d_q.release();
    d_out.release();
    return err;
  }
  err = device_buf_alloc(&d_v_full, v_full_bytes);
  if (!err.ok()) {
    d_q.release();
    d_out.release();
    d_k_full.release();
    return err;
  }

  // Pack past + new K/V into contiguous device buffers.
  auto upload_row = [&](const kernels::TensorView& t, float* dst, std::size_t bytes) -> Error {
    if (cuda_is_device_ptr(t.data)) {
      return cuda_status(cudaMemcpy(dst, t.data, bytes, cudaMemcpyDeviceToDevice), "D2D pack");
    }
    return cuda_status(cudaMemcpy(dst, t.data, bytes, cudaMemcpyHostToDevice), "H2D pack");
  };

  for (std::int64_t b = 0; b < batch; ++b) {
    if (past_seq > 0) {
      const std::size_t past_k_bytes =
          static_cast<std::size_t>(past_seq * k_dim) * sizeof(float);
      const std::size_t past_v_bytes =
          static_cast<std::size_t>(past_seq * v_dim) * sizeof(float);
      const char* pk =
          static_cast<const char*>(past_k->data) +
          static_cast<std::size_t>(b * past_seq * k_dim) * sizeof(float);
      const char* pv =
          static_cast<const char*>(past_v->data) +
          static_cast<std::size_t>(b * past_seq * v_dim) * sizeof(float);
      if (cuda_is_device_ptr(past_k->data)) {
        err = cuda_status(cudaMemcpy(d_k_full.p + b * kv_seq * k_dim, pk, past_k_bytes,
                                     cudaMemcpyDeviceToDevice),
                          "past K D2D");
      } else {
        err = cuda_status(cudaMemcpy(d_k_full.p + b * kv_seq * k_dim, pk, past_k_bytes,
                                     cudaMemcpyHostToDevice),
                          "past K H2D");
      }
      if (!err.ok()) break;
      if (cuda_is_device_ptr(past_v->data)) {
        err = cuda_status(cudaMemcpy(d_v_full.p + b * kv_seq * v_dim, pv, past_v_bytes,
                                     cudaMemcpyDeviceToDevice),
                          "past V D2D");
      } else {
        err = cuda_status(cudaMemcpy(d_v_full.p + b * kv_seq * v_dim, pv, past_v_bytes,
                                     cudaMemcpyHostToDevice),
                          "past V H2D");
      }
      if (!err.ok()) break;
    }
    const std::size_t new_k_bytes = static_cast<std::size_t>(k_dim) * sizeof(float);
    const std::size_t new_v_bytes = static_cast<std::size_t>(v_dim) * sizeof(float);
    kernels::TensorView k_slice = inputs[1];
    kernels::TensorView v_slice = inputs[2];
    // Point at batch row for rank-2 [B,dim] or [B,1,dim].
    const std::size_t k_off = static_cast<std::size_t>(b * k_dim) * sizeof(float);
    const std::size_t v_off = static_cast<std::size_t>(b * v_dim) * sizeof(float);
    k_slice.data = static_cast<char*>(inputs[1].data) + k_off;
    k_slice.nbytes = new_k_bytes;
    v_slice.data = static_cast<char*>(inputs[2].data) + v_off;
    v_slice.nbytes = new_v_bytes;
    err = upload_row(k_slice, d_k_full.p + (b * kv_seq + past_seq) * k_dim, new_k_bytes);
    if (!err.ok()) break;
    err = upload_row(v_slice, d_v_full.p + (b * kv_seq + past_seq) * v_dim, new_v_bytes);
    if (!err.ok()) break;
  }
  if (!err.ok()) {
    d_q.release();
    d_out.release();
    d_k_full.release();
    d_v_full.release();
    return err;
  }

  err = cuda_attention_decode_f32(q_dev, d_k_full.p, d_v_full.p, out_dev, batch, kv_seq, q_dim,
                                  k_dim, v_dim, out_dim, num_heads, num_kv_heads, head_dim,
                                  v_head_dim, scale, logit_softcap);
  if (err.ok() && need_d2h) {
    err = cuda_status(
        cudaMemcpy((*outputs)[0].data, out_dev, (*outputs)[0].nbytes, cudaMemcpyDeviceToHost),
        "attn D2H");
  }

  d_q.release();
  d_out.release();
  d_k_full.release();
  d_v_full.release();
  if (!err.ok()) {
    return err;
  }

  // Present K/V explicit outputs not implemented on CUDA decode path.
  if (outputs->size() >= 3) {
    return Error::make(ErrorCode::NotImplemented,
                       "CUDA Attention present_k/v outputs not implemented");
  }

  if (use_kv && cache != nullptr && layer_id >= 0) {
    // Append uses host K/V; if device-resident, stage through temporary host copies.
    if (cuda_is_device_ptr(inputs[1].data) || cuda_is_device_ptr(inputs[2].data)) {
      std::vector<float> hk(static_cast<std::size_t>(inputs[1].numel()));
      std::vector<float> hv(static_cast<std::size_t>(inputs[2].numel()));
      if (cuda_is_device_ptr(inputs[1].data)) {
        err = cuda_status(cudaMemcpy(hk.data(), inputs[1].data, inputs[1].nbytes,
                                     cudaMemcpyDeviceToHost),
                          "attn K for cache");
      } else {
        std::memcpy(hk.data(), inputs[1].data, inputs[1].nbytes);
      }
      if (!err.ok()) return err;
      if (cuda_is_device_ptr(inputs[2].data)) {
        err = cuda_status(cudaMemcpy(hv.data(), inputs[2].data, inputs[2].nbytes,
                                     cudaMemcpyDeviceToHost),
                          "attn V for cache");
      } else {
        std::memcpy(hv.data(), inputs[2].data, inputs[2].nbytes);
      }
      if (!err.ok()) return err;
      kernels::TensorView k_host = inputs[1];
      kernels::TensorView v_host = inputs[2];
      k_host.data = hk.data();
      v_host.data = hv.data();
      return cache->append_layer(layer_id, k_host, v_host);
    }
    return cache->append_layer(layer_id, inputs[1], inputs[2]);
  }
  return Error::success();
}

}  // namespace

bool cuda_compiled() noexcept { return true; }

const char* cuda_capability_details() noexcept {
  return "CUDA native: MatMul(cuBLASLt+cublasSgemm fallback),Add,Mul,RMSNorm,Softmax,Silu,"
         "Attention(decode seq=1 staged/on-device),RoPE(staged/on-device); "
         "cuBLASLt + dual-stream; host_fallback: MLAExpandK,Embedding,Attention prefill,others";
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
  if (op_name == "RoPE") {
    return dispatch_rope(inputs, outputs, attrs);
  }
  if (op_name == "Attention") {
    return dispatch_attention(inputs, outputs, attrs);
  }
  if (op_name == "MLAExpandK") {
    return Error::make(ErrorCode::NotImplemented,
                       "CUDA MLAExpandK not implemented (CPU/host staging)");
  }
  return Error::make(ErrorCode::NotImplemented,
                     "CUDA native op not implemented: " + op_name);
}

bool cuda_probe_device() noexcept {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool cuda_is_device_ptr(const void* p) noexcept {
  if (p == nullptr) {
    return false;
  }
  cudaPointerAttributes attr{};
  const cudaError_t st = cudaPointerGetAttributes(&attr, p);
  if (st != cudaSuccess) {
    (void)cudaGetLastError();  // clear sticky error from host pointer probe
    return false;
  }
#if CUDART_VERSION >= 10000
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else
  return attr.memoryType == cudaMemoryTypeDevice;
#endif
}

}  // namespace native
}  // namespace backends
}  // namespace uaii
