// Apple Metal device memory + on-device compute (MatMul/Add/RMSNorm).



#include "native_stubs.hpp"



#import <Foundation/Foundation.h>

#import <Metal/Metal.h>



#include <cmath>

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

id<MTLDevice> g_device = nil;

id<MTLCommandQueue> g_queue = nil;

std::unordered_map<void*, id<MTLBuffer>> g_buffers;

bool g_pipelines_ready = false;



id<MTLComputePipelineState> g_pipe_add = nil;

id<MTLComputePipelineState> g_pipe_matmul = nil;

id<MTLComputePipelineState> g_pipe_rmsnorm = nil;



static const char* kMetalLibrarySource = R"(

#include <metal_stdlib>

using namespace metal;



kernel void uaii_add_f32(device const float* a [[buffer(0)]],

                         device const float* b [[buffer(1)]],

                         device float* c [[buffer(2)]],

                         constant uint& n [[buffer(3)]],

                         uint i [[thread_position_in_grid]]) {

  if (i >= n) return;

  c[i] = a[i] + b[i];

}



kernel void uaii_matmul_f32(device const float* a [[buffer(0)]],

                            device const float* b [[buffer(1)]],

                            device float* c [[buffer(2)]],

                            constant uint3& dims [[buffer(3)]],

                            uint2 gid [[thread_position_in_grid]]) {

  const uint row = gid.y;

  const uint col = gid.x;

  const uint m = dims.x;

  const uint n = dims.y;

  const uint k = dims.z;

  if (row >= m || col >= n) return;

  float sum = 0.0f;

  for (uint t = 0; t < k; ++t) {

    sum += a[row * k + t] * b[t * n + col];

  }

  c[row * n + col] = sum;

}



kernel void uaii_rmsnorm_f32(device const float* x [[buffer(0)]],

                             device const float* weight [[buffer(1)]],

                             device float* y [[buffer(2)]],

                             constant float4& params [[buffer(3)]],

                             uint row [[thread_position_in_grid]]) {

  const uint rows = as_type<uint>(params.x);

  const uint cols = as_type<uint>(params.y);

  const float eps = params.z;

  const float has_w = params.w;

  if (row >= rows) return;

  device const float* row_x = x + row * cols;

  device float* row_y = y + row * cols;

  float acc = 0.0f;

  for (uint c = 0; c < cols; ++c) {

    const float v = row_x[c];

    acc += v * v;

  }

  const float inv = rsqrt(acc / float(cols) + eps);

  if (has_w > 0.5f) {

    for (uint c = 0; c < cols; ++c) {

      row_y[c] = row_x[c] * inv * weight[c];

    }

  } else {

    for (uint c = 0; c < cols; ++c) {

      row_y[c] = row_x[c] * inv;

    }

  }

}

)";



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



Error require_f32(const kernels::TensorView& t, const char* name) {

  if (t.dtype != DType::F32 || t.data == nullptr) {

    return Error::make(ErrorCode::InvalidArgument,

                       std::string("Metal ") + name + " requires f32 data");

  }

  return Error::success();

}



id<MTLBuffer> buffer_for_ptr(void* ptr) {

  auto it = g_buffers.find(ptr);

  if (it == g_buffers.end()) {

    return nil;

  }

  return it->second;

}



Error build_pipelines() {

  if (g_device == nil) {

    return Error::make(ErrorCode::Internal, "Metal device missing");

  }

  NSError* err = nil;

  id<MTLLibrary> lib =

      [g_device newLibraryWithSource:[NSString stringWithUTF8String:kMetalLibrarySource]

                             options:nil

                               error:&err];

  if (lib == nil) {

    return Error::make(ErrorCode::Internal,

                       std::string("Metal shader compile failed: ") +

                           (err ? [[err localizedDescription] UTF8String] : "unknown"));

  }



  auto make_pipe = [&](const char* name) -> id<MTLComputePipelineState> {

    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];

    if (fn == nil) {

      return nil;

    }

    return [g_device newComputePipelineStateWithFunction:fn error:&err];

  };



  g_pipe_add = make_pipe("uaii_add_f32");

  g_pipe_matmul = make_pipe("uaii_matmul_f32");

  g_pipe_rmsnorm = make_pipe("uaii_rmsnorm_f32");

  if (g_pipe_add == nil || g_pipe_matmul == nil || g_pipe_rmsnorm == nil) {

    g_pipe_add = nil;

    g_pipe_matmul = nil;

    g_pipe_rmsnorm = nil;

    return Error::make(ErrorCode::Internal, "Metal pipeline creation failed");

  }

  g_pipelines_ready = true;

  return Error::success();

}



Error run_compute(id<MTLComputePipelineState> pipe,

                  const std::vector<std::pair<id<MTLBuffer>, NSUInteger>>& buffers,

                  MTLSize grid, MTLSize tpg) {

  if (pipe == nil || g_queue == nil) {

    return Error::make(ErrorCode::Internal, "Metal pipeline/queue missing");

  }

  id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

  [enc setComputePipelineState:pipe];

  for (std::size_t i = 0; i < buffers.size(); ++i) {

    [enc setBuffer:buffers[i].first offset:buffers[i].second atIndex:static_cast<NSUInteger>(i)];

  }

  [enc dispatchThreads:grid threadsPerThreadgroup:tpg];

  [enc endEncoding];

  [cmd commit];

  [cmd waitUntilCompleted];

  if (cmd.status != MTLCommandBufferStatusCompleted) {

    return Error::make(ErrorCode::Internal, "Metal command buffer failed");

  }

  return Error::success();

}



Error dispatch_add(const std::vector<kernels::TensorView>& inputs,

                   std::vector<kernels::TensorView>* outputs) {

  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {

    return Error::make(ErrorCode::InvalidArgument, "Metal Add arity");

  }

  Error err = require_f32(inputs[0], "A");

  if (!err.ok()) return err;

  err = require_f32(inputs[1], "B");

  if (!err.ok()) return err;

  err = require_f32((*outputs)[0], "C");

  if (!err.ok()) return err;

  if (inputs[0].numel() != inputs[1].numel() ||

      inputs[0].numel() != (*outputs)[0].numel()) {

    return Error::make(ErrorCode::InvalidArgument, "Metal Add size");

  }



  id<MTLBuffer> ba = buffer_for_ptr(inputs[0].data);

  id<MTLBuffer> bb = buffer_for_ptr(inputs[1].data);

  id<MTLBuffer> bc = buffer_for_ptr((*outputs)[0].data);

  if (ba == nil || bb == nil || bc == nil) {

    return Error::make(ErrorCode::InvalidArgument, "Metal Add buffer not owned");

  }



  const uint32_t n = static_cast<uint32_t>(inputs[0].numel());

  id<MTLBuffer> param = [g_device newBufferWithBytes:&n

                                                length:sizeof(n)

                                               options:MTLResourceStorageModeShared];

  return run_compute(g_pipe_add,

                     {{ba, 0}, {bb, 0}, {bc, 0}, {param, 0}},

                     MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1));

}



Error dispatch_matmul(const std::vector<kernels::TensorView>& inputs,

                      std::vector<kernels::TensorView>* outputs,

                      const std::vector<ir::Attribute>& attrs) {

  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {

    return Error::make(ErrorCode::InvalidArgument, "Metal MatMul arity");

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

    return Error::make(ErrorCode::InvalidArgument, "Metal MatMul expects rank-2");

  }

  const bool ta = attr_bool(attrs, "transpose_a", false);

  const bool tb = attr_bool(attrs, "transpose_b", false);

  if (ta || tb) {

    return Error::make(ErrorCode::NotImplemented, "Metal MatMul transpose not supported yet");

  }

  const uint32_t m = static_cast<uint32_t>(a.dim(0));

  const uint32_t k = static_cast<uint32_t>(a.dim(1));

  const uint32_t n = static_cast<uint32_t>(b.dim(1));

  if (b.dim(0) != static_cast<std::int64_t>(k) || c.dim(0) != static_cast<std::int64_t>(m) ||

      c.dim(1) != static_cast<std::int64_t>(n)) {

    return Error::make(ErrorCode::InvalidArgument, "Metal MatMul shape mismatch");

  }



  id<MTLBuffer> ba = buffer_for_ptr(a.data);

  id<MTLBuffer> bb = buffer_for_ptr(b.data);

  id<MTLBuffer> bc = buffer_for_ptr(c.data);

  if (ba == nil || bb == nil || bc == nil) {

    return Error::make(ErrorCode::InvalidArgument, "Metal MatMul buffer not owned");

  }



  const uint32_t dims[3] = {m, n, k};

  id<MTLBuffer> param = [g_device newBufferWithBytes:dims

                                                length:sizeof(dims)

                                               options:MTLResourceStorageModeShared];

  return run_compute(g_pipe_matmul, {{ba, 0}, {bb, 0}, {bc, 0}, {param, 0}},

                     MTLSizeMake(n, m, 1), MTLSizeMake(16, 16, 1));

}



Error dispatch_rmsnorm(const std::vector<kernels::TensorView>& inputs,

                       std::vector<kernels::TensorView>* outputs,

                       const std::vector<ir::Attribute>& attrs) {

  if (inputs.empty() || outputs == nullptr || outputs->size() != 1) {

    return Error::make(ErrorCode::InvalidArgument, "Metal RMSNorm arity");

  }

  Error err = require_f32(inputs[0], "x");

  if (!err.ok()) return err;

  err = require_f32((*outputs)[0], "y");

  if (!err.ok()) return err;

  if (inputs[0].rank < 1) {

    return Error::make(ErrorCode::InvalidArgument, "Metal RMSNorm rank");

  }

  const std::int64_t cols = inputs[0].dim(inputs[0].rank - 1);

  std::int64_t rows = 1;

  for (std::size_t i = 0; i + 1 < inputs[0].rank; ++i) {

    rows *= inputs[0].dim(i);

  }

  if (rows * cols != static_cast<std::int64_t>(inputs[0].numel()) ||

      inputs[0].numel() != (*outputs)[0].numel()) {

    return Error::make(ErrorCode::InvalidArgument, "Metal RMSNorm size");

  }



  id<MTLBuffer> bx = buffer_for_ptr(inputs[0].data);

  id<MTLBuffer> by = buffer_for_ptr((*outputs)[0].data);

  if (bx == nil || by == nil) {

    return Error::make(ErrorCode::InvalidArgument, "Metal RMSNorm buffer not owned");

  }



  id<MTLBuffer> bw = nil;

  float has_w = 0.0f;

  if (inputs.size() > 1 && inputs[1].data != nullptr) {

    err = require_f32(inputs[1], "weight");

    if (!err.ok()) return err;

    if (static_cast<std::int64_t>(inputs[1].numel()) != cols) {

      return Error::make(ErrorCode::InvalidArgument, "Metal RMSNorm weight");

    }

    bw = buffer_for_ptr(inputs[1].data);

    if (bw == nil) {

      return Error::make(ErrorCode::InvalidArgument, "Metal RMSNorm weight buffer not owned");

    }

    has_w = 1.0f;

  }

  if (bw == nil) {

    bw = [g_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

  }



  const float eps = attr_float(attrs, "eps", 1e-5f);

  const float params[4] = {static_cast<float>(rows), static_cast<float>(cols), eps, has_w};

  id<MTLBuffer> param = [g_device newBufferWithBytes:params

                                                length:sizeof(params)

                                               options:MTLResourceStorageModeShared];

  return run_compute(g_pipe_rmsnorm, {{bx, 0}, {bw, 0}, {by, 0}, {param, 0}},

                     MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1), MTLSizeMake(256, 1, 1));

}



}  // namespace



bool metal_compiled() noexcept { return true; }



const char* metal_capability_details() noexcept {

  if (g_pipelines_ready) {

    return "Metal native: MatMul,Add,RMSNorm on-device (MSL); host_fallback: "

           "Softmax,Silu,Attention,others";

  }

  return "Metal native buffers; host_fallback: MatMul,Add,RMSNorm,Softmax,Silu,Attention,others";

}



Error metal_init() {

  std::lock_guard<std::mutex> lock(g_mu);

  if (g_device != nil) {

    return Error::success();

  }

  g_device = MTLCreateSystemDefaultDevice();

  if (g_device == nil) {

    return Error::make(ErrorCode::NotFound, "no Metal device");

  }

  g_queue = [g_device newCommandQueue];

  if (g_queue == nil) {

    g_device = nil;

    return Error::make(ErrorCode::Internal, "Metal command queue create failed");

  }

  g_pipelines_ready = false;

  Error err = build_pipelines();

  if (!err.ok()) {

    return err;

  }

  return Error::success();

}



void metal_shutdown() noexcept {

  std::lock_guard<std::mutex> lock(g_mu);

  g_pipe_add = nil;

  g_pipe_matmul = nil;

  g_pipe_rmsnorm = nil;

  g_pipelines_ready = false;

  g_buffers.clear();

  g_queue = nil;

  g_device = nil;

}



Error metal_allocate(std::size_t bytes, void** out_ptr) {

  if (out_ptr == nullptr) {

    return Error::make(ErrorCode::InvalidArgument, "metal_allocate out null");

  }

  std::lock_guard<std::mutex> lock(g_mu);

  if (g_device == nil) {

    return Error::make(ErrorCode::InvalidArgument, "Metal not initialized");

  }

  const NSUInteger len = bytes == 0 ? 1 : static_cast<NSUInteger>(bytes);

  id<MTLBuffer> buf = [g_device newBufferWithLength:len

                                             options:MTLResourceStorageModeShared];

  if (buf == nil) {

    return Error::make(ErrorCode::Internal, "Metal buffer alloc failed");

  }

  void* p = [buf contents];

  g_buffers[p] = buf;

  *out_ptr = p;

  return Error::success();

}



Error metal_free(void* ptr) noexcept {

  if (ptr == nullptr) {

    return Error::success();

  }

  std::lock_guard<std::mutex> lock(g_mu);

  g_buffers.erase(ptr);

  return Error::success();

}



Error metal_copy_h2d(const void* host, void* device, std::size_t bytes) {

  if (bytes == 0) return Error::success();

  if (host == nullptr || device == nullptr) {

    return Error::make(ErrorCode::InvalidArgument, "metal copy_h2d null");

  }

  std::memcpy(device, host, bytes);

  return Error::success();

}



Error metal_copy_d2h(const void* device, void* host, std::size_t bytes) {

  if (bytes == 0) return Error::success();

  if (host == nullptr || device == nullptr) {

    return Error::make(ErrorCode::InvalidArgument, "metal copy_d2h null");

  }

  std::memcpy(host, device, bytes);

  return Error::success();

}



Error metal_copy_d2d(const void* src, void* dst, std::size_t bytes) {

  if (bytes == 0) return Error::success();

  if (src == nullptr || dst == nullptr) {

    return Error::make(ErrorCode::InvalidArgument, "metal copy_d2d null");

  }

  std::memcpy(dst, src, bytes);

  return Error::success();

}



Error metal_synchronize() {

  return Error::success();

}



Error metal_dispatch(const std::string& op_name, const std::string&,

                     const std::vector<kernels::TensorView>& inputs,

                     std::vector<kernels::TensorView>* outputs,

                     const std::vector<ir::Attribute>& attrs) {

  if (!g_pipelines_ready) {

    return Error::make(ErrorCode::NotImplemented,

                       "Metal compute pipelines not ready (use host_fallback)");

  }

  if (op_name == "Add") {

    return dispatch_add(inputs, outputs);

  }

  if (op_name == "MatMul" || op_name == "MatMulRelu") {

    Error err = dispatch_matmul(inputs, outputs, attrs);

    if (!err.ok() || op_name == "MatMul") {

      return err;

    }

    return Error::make(ErrorCode::NotImplemented, "Metal MatMulRelu not implemented");

  }

  if (op_name == "RMSNorm") {

    return dispatch_rmsnorm(inputs, outputs, attrs);

  }

  return Error::make(ErrorCode::NotImplemented,

                     "Metal native op not implemented: " + op_name);

}



}  // namespace native

}  // namespace backends

}  // namespace uaii

