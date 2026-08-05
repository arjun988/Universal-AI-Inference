#include "native_stubs.hpp"
#include "vulkan_spirv.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#define UAII_HAVE_VULKAN_HEADERS 1
#include <vulkan/vulkan.h>
#endif
#endif
namespace uaii {
namespace backends {
namespace native {
namespace {
#if defined(UAII_HAVE_VULKAN_HEADERS)
std::mutex g_mu;
bool g_inited = false;
bool g_add_pipeline_ready = false;
VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_queue_family = 0;
VkCommandPool g_pool = VK_NULL_HANDLE;
VkDescriptorSetLayout g_add_dsl = VK_NULL_HANDLE;
VkPipelineLayout g_add_layout = VK_NULL_HANDLE;
VkPipeline g_add_pipe = VK_NULL_HANDLE;
VkDescriptorPool g_add_pool = VK_NULL_HANDLE;
VkDescriptorSet g_add_set = VK_NULL_HANDLE;
VkBuffer g_param_buf = VK_NULL_HANDLE;
VkDeviceMemory g_param_mem = VK_NULL_HANDLE;
void* g_param_mapped = nullptr;
struct AllocRec {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  void* mapped = nullptr;
  std::size_t bytes = 0;
};
std::unordered_map<void*, AllocRec> g_allocs;
Error vk_err(VkResult r, const char* what) {
  if (r == VK_SUCCESS) {
    return Error::success();
  }
  return Error::make(ErrorCode::Internal, std::string("Vulkan ") + what + " failed");
}
uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(g_phys, &mem_props);
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  return UINT32_MAX;
}
AllocRec* rec_for_ptr(void* ptr) {
  auto it = g_allocs.find(ptr);
  if (it == g_allocs.end()) {
    return nullptr;
  }
  return &it->second;
}
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
                       std::string("Vulkan ") + name + " requires f32 data");
  }
  return Error::success();
}
Error create_shader_module(const std::uint32_t* code, std::size_t words,
                           VkShaderModule* out) {
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = words * sizeof(std::uint32_t);
  ci.pCode = code;
  return vk_err(vkCreateShaderModule(g_device, &ci, nullptr, out), "CreateShaderModule");
}
Error build_add_pipeline() {
  VkShaderModule shader = VK_NULL_HANDLE;
  Error err = create_shader_module(vulkan_spirv::kAddSpv, vulkan_spirv::kAddSpv_words, &shader);
  if (!err.ok()) {
    return err;
  }
  VkDescriptorSetLayoutBinding binds[4]{};
  for (uint32_t i = 0; i < 3; ++i) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  binds[3].binding = 3;
  binds[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  binds[3].descriptorCount = 1;
  binds[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dsl_ci{};
  dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsl_ci.bindingCount = 4;
  dsl_ci.pBindings = binds;
  err = vk_err(vkCreateDescriptorSetLayout(g_device, &dsl_ci, nullptr, &g_add_dsl),
               "CreateDescriptorSetLayout");
  if (!err.ok()) {
    vkDestroyShaderModule(g_device, shader, nullptr);
    return err;
  }
  VkPipelineLayoutCreateInfo pl_ci{};
  pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pl_ci.setLayoutCount = 1;
  pl_ci.pSetLayouts = &g_add_dsl;
  err = vk_err(vkCreatePipelineLayout(g_device, &pl_ci, nullptr, &g_add_layout),
               "CreatePipelineLayout");
  if (!err.ok()) {
    vkDestroyShaderModule(g_device, shader, nullptr);
    return err;
  }
  VkComputePipelineCreateInfo pipe_ci{};
  pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipe_ci.stage.module = shader;
  pipe_ci.stage.pName = "main";
  pipe_ci.layout = g_add_layout;
  err = vk_err(vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &pipe_ci, nullptr,
                                        &g_add_pipe),
               "CreateComputePipelines");
  vkDestroyShaderModule(g_device, shader, nullptr);
  if (!err.ok()) {
    return err;
  }
  VkDescriptorPoolSize sizes[2]{};
  sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sizes[0].descriptorCount = 3;
  sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sizes[1].descriptorCount = 1;
  VkDescriptorPoolCreateInfo pool_ci{};
  pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_ci.maxSets = 1;
  pool_ci.poolSizeCount = 2;
  pool_ci.pPoolSizes = sizes;
  err = vk_err(vkCreateDescriptorPool(g_device, &pool_ci, nullptr, &g_add_pool),
               "CreateDescriptorPool");
  if (!err.ok()) {
    return err;
  }
  VkDescriptorSetAllocateInfo set_ai{};
  set_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_ai.descriptorPool = g_add_pool;
  set_ai.descriptorSetCount = 1;
  set_ai.pSetLayouts = &g_add_dsl;
  err = vk_err(vkAllocateDescriptorSets(g_device, &set_ai, &g_add_set), "AllocateDescriptorSets");
  if (!err.ok()) {
    return err;
  }
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = 256;
  bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  err = vk_err(vkCreateBuffer(g_device, &bci, nullptr, &g_param_buf), "CreateBuffer param");
  if (!err.ok()) {
    return err;
  }
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(g_device, g_param_buf, &req);
  const uint32_t mem_type =
      find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mem_type == UINT32_MAX) {
    return Error::make(ErrorCode::Internal, "Vulkan param memory type missing");
  }
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mem_type;
  err = vk_err(vkAllocateMemory(g_device, &mai, nullptr, &g_param_mem), "AllocateMemory param");
  if (!err.ok()) {
    return err;
  }
  err = vk_err(vkBindBufferMemory(g_device, g_param_buf, g_param_mem, 0), "BindBufferMemory param");
  if (!err.ok()) {
    return err;
  }
  err = vk_err(vkMapMemory(g_device, g_param_mem, 0, req.size, 0, &g_param_mapped),
               "MapMemory param");
  if (!err.ok()) {
    return err;
  }
  g_add_pipeline_ready = true;
  return Error::success();
}
Error submit_compute(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                     std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cb_ai{};
  cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cb_ai.commandPool = g_pool;
  cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cb_ai.commandBufferCount = 1;
  Error err = vk_err(vkAllocateCommandBuffers(g_device, &cb_ai, &cmd), "AllocateCommandBuffers");
  if (!err.ok()) {
    return err;
  }
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  err = vk_err(vkBeginCommandBuffer(cmd, &bi), "BeginCommandBuffer");
  if (!err.ok()) {
    vkFreeCommandBuffers(g_device, g_pool, 1, &cmd);
    return err;
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
  vkCmdDispatch(cmd, gx, gy, gz);
  err = vk_err(vkEndCommandBuffer(cmd), "EndCommandBuffer");
  if (!err.ok()) {
    vkFreeCommandBuffers(g_device, g_pool, 1, &cmd);
    return err;
  }
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  err = vk_err(vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE), "QueueSubmit");
  vkFreeCommandBuffers(g_device, g_pool, 1, &cmd);
  if (!err.ok()) {
    return err;
  }
  return vk_err(vkQueueWaitIdle(g_queue), "QueueWaitIdle");
}
Error dispatch_add_gpu(const std::vector<kernels::TensorView>& inputs,
                       std::vector<kernels::TensorView>* outputs) {
  if (!g_add_pipeline_ready) {
    return Error::make(ErrorCode::NotImplemented, "Vulkan Add pipeline not ready");
  }
  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan Add arity");
  }
  Error err = require_f32(inputs[0], "A");
  if (!err.ok()) return err;
  err = require_f32(inputs[1], "B");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "C");
  if (!err.ok()) return err;
  if (inputs[0].numel() != inputs[1].numel() ||
      inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan Add size");
  }
  AllocRec* ra = rec_for_ptr(inputs[0].data);
  AllocRec* rb = rec_for_ptr(inputs[1].data);
  AllocRec* rc = rec_for_ptr((*outputs)[0].data);
  if (ra == nullptr || rb == nullptr || rc == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan Add buffer not owned");
  }
  const std::uint32_t n = static_cast<std::uint32_t>(inputs[0].numel());
  std::memcpy(g_param_mapped, &n, sizeof(n));
  VkDescriptorBufferInfo infos[4]{};
  infos[0].buffer = ra->buffer;
  infos[0].offset = 0;
  infos[0].range = VK_WHOLE_SIZE;
  infos[1].buffer = rb->buffer;
  infos[1].offset = 0;
  infos[1].range = VK_WHOLE_SIZE;
  infos[2].buffer = rc->buffer;
  infos[2].offset = 0;
  infos[2].range = VK_WHOLE_SIZE;
  infos[3].buffer = g_param_buf;
  infos[3].offset = 0;
  infos[3].range = sizeof(n);
  VkWriteDescriptorSet writes[4]{};
  for (int i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = g_add_set;
    writes[i].dstBinding = static_cast<uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[3].dstSet = g_add_set;
  writes[3].dstBinding = 3;
  writes[3].descriptorCount = 1;
  writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes[3].pBufferInfo = &infos[3];
  vkUpdateDescriptorSets(g_device, 4, writes, 0, nullptr);
  const std::uint32_t groups = (n + 255u) / 256u;
  return submit_compute(g_add_pipe, g_add_layout, g_add_set, groups, 1, 1);
}
Error dispatch_add_mapped(const std::vector<kernels::TensorView>& inputs,
                          std::vector<kernels::TensorView>* outputs) {
  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan Add arity");
  }
  Error err = require_f32(inputs[0], "A");
  if (!err.ok()) return err;
  err = require_f32(inputs[1], "B");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "C");
  if (!err.ok()) return err;
  if (inputs[0].numel() != inputs[1].numel() ||
      inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan Add size");
  }
  if (rec_for_ptr(inputs[0].data) == nullptr || rec_for_ptr(inputs[1].data) == nullptr ||
      rec_for_ptr((*outputs)[0].data) == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan Add buffer not owned");
  }
  const float* a = inputs[0].f32();
  const float* b = inputs[1].f32();
  float* c = (*outputs)[0].f32();
  for (std::size_t i = 0; i < inputs[0].numel(); ++i) {
    c[i] = a[i] + b[i];
  }
  return Error::success();
}
Error dispatch_matmul_mapped(const std::vector<kernels::TensorView>& inputs,
                             std::vector<kernels::TensorView>* outputs,
                             const std::vector<ir::Attribute>& attrs) {
  if (inputs.size() != 2 || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan MatMul arity");
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
    return Error::make(ErrorCode::InvalidArgument, "Vulkan MatMul expects rank-2");
  }
  const bool ta = attr_bool(attrs, "transpose_a", false);
  const bool tb = attr_bool(attrs, "transpose_b", false);
  if (ta || tb) {
    return Error::make(ErrorCode::NotImplemented, "Vulkan MatMul transpose not supported");
  }
  const std::int64_t m = a.dim(0);
  const std::int64_t k = a.dim(1);
  const std::int64_t n = b.dim(1);
  if (b.dim(0) != k || c.dim(0) != m || c.dim(1) != n) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan MatMul shape mismatch");
  }
  if (rec_for_ptr(a.data) == nullptr || rec_for_ptr(b.data) == nullptr ||
      rec_for_ptr(c.data) == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan MatMul buffer not owned");
  }
  float* af = a.f32();
  float* bf = b.f32();
  float* cf = c.f32();
  for (std::int64_t row = 0; row < m; ++row) {
    for (std::int64_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (std::int64_t t = 0; t < k; ++t) {
        sum += af[row * k + t] * bf[t * n + col];
      }
      cf[row * n + col] = sum;
    }
  }
  return Error::success();
}
Error dispatch_rmsnorm_mapped(const std::vector<kernels::TensorView>& inputs,
                                std::vector<kernels::TensorView>* outputs,
                                const std::vector<ir::Attribute>& attrs) {
  if (inputs.empty() || outputs == nullptr || outputs->size() != 1) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan RMSNorm arity");
  }
  Error err = require_f32(inputs[0], "x");
  if (!err.ok()) return err;
  err = require_f32((*outputs)[0], "y");
  if (!err.ok()) return err;
  if (inputs[0].rank < 1) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan RMSNorm rank");
  }
  const std::int64_t cols = inputs[0].dim(inputs[0].rank - 1);
  std::int64_t rows = 1;
  for (std::size_t i = 0; i + 1 < inputs[0].rank; ++i) {
    rows *= inputs[0].dim(i);
  }
  if (rows * cols != static_cast<std::int64_t>(inputs[0].numel()) ||
      inputs[0].numel() != (*outputs)[0].numel()) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan RMSNorm size");
  }
  if (rec_for_ptr(inputs[0].data) == nullptr || rec_for_ptr((*outputs)[0].data) == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan RMSNorm buffer not owned");
  }
  const float* x = inputs[0].f32();
  float* y = (*outputs)[0].f32();
  const float* w = nullptr;
  if (inputs.size() > 1 && inputs[1].data != nullptr) {
    err = require_f32(inputs[1], "weight");
    if (!err.ok()) return err;
    if (static_cast<std::int64_t>(inputs[1].numel()) != cols) {
      return Error::make(ErrorCode::InvalidArgument, "Vulkan RMSNorm weight");
    }
    if (rec_for_ptr(inputs[1].data) == nullptr) {
      return Error::make(ErrorCode::InvalidArgument, "Vulkan RMSNorm weight buffer not owned");
    }
    w = inputs[1].f32();
  }
  const float eps = attr_float(attrs, "eps", 1e-5f);
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* row = x + r * cols;
    float* out = y + r * cols;
    float acc = 0.0f;
    for (std::int64_t c = 0; c < cols; ++c) {
      const float v = row[c];
      acc += v * v;
    }
    const float inv = 1.0f / std::sqrt(acc / static_cast<float>(cols) + eps);
    if (w != nullptr) {
      for (std::int64_t c = 0; c < cols; ++c) {
        out[c] = row[c] * inv * w[c];
      }
    } else {
      for (std::int64_t c = 0; c < cols; ++c) {
        out[c] = row[c] * inv;
      }
    }
  }
  return Error::success();
}
void destroy_add_pipeline() {
  g_add_pipeline_ready = false;
  if (g_param_mapped != nullptr && g_param_mem != VK_NULL_HANDLE) {
    vkUnmapMemory(g_device, g_param_mem);
    g_param_mapped = nullptr;
  }
  if (g_param_buf != VK_NULL_HANDLE) {
    vkDestroyBuffer(g_device, g_param_buf, nullptr);
    g_param_buf = VK_NULL_HANDLE;
  }
  if (g_param_mem != VK_NULL_HANDLE) {
    vkFreeMemory(g_device, g_param_mem, nullptr);
    g_param_mem = VK_NULL_HANDLE;
  }
  if (g_add_pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(g_device, g_add_pool, nullptr);
    g_add_pool = VK_NULL_HANDLE;
  }
  g_add_set = VK_NULL_HANDLE;
  if (g_add_pipe != VK_NULL_HANDLE) {
    vkDestroyPipeline(g_device, g_add_pipe, nullptr);
    g_add_pipe = VK_NULL_HANDLE;
  }
  if (g_add_layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(g_device, g_add_layout, nullptr);
    g_add_layout = VK_NULL_HANDLE;
  }
  if (g_add_dsl != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(g_device, g_add_dsl, nullptr);
    g_add_dsl = VK_NULL_HANDLE;
  }
}
#endif
}  // namespace
bool vulkan_compiled() noexcept { return true; }
const char* vulkan_capability_details() noexcept {
#if defined(UAII_HAVE_VULKAN_HEADERS)
  if (g_add_pipeline_ready) {
    return "Vulkan native buffers; Add(VkCompute); MatMul/RMSNorm CPU-on-mapped "
           "(host_fallback=true); Softmax,Silu,Attention host";
  }
  return "Vulkan native host-visible buffers; host_fallback=true for MatMul,Add,"
         "RMSNorm,Softmax,Silu,Attention (compute shaders pending)";
#else
  return "Vulkan headers unavailable — host_fallback: all ops";
#endif
}
Error vulkan_init() {
#if !defined(UAII_HAVE_VULKAN_HEADERS)
  return Error::make(ErrorCode::NotImplemented,
                     "Vulkan headers not found (install Vulkan SDK / vulkan.h)");
#else
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_inited) {
    return Error::success();
  }
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "uaii";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  Error err = vk_err(vkCreateInstance(&ici, nullptr, &g_instance), "CreateInstance");
  if (!err.ok()) {
    return err;
  }
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(g_instance, &count, nullptr);
  if (count == 0) {
    vkDestroyInstance(g_instance, nullptr);
    g_instance = VK_NULL_HANDLE;
    return Error::make(ErrorCode::NotFound, "no Vulkan physical devices");
  }
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(g_instance, &count, devices.data());
  g_phys = devices[0];
  uint32_t qcount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qcount, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qcount);
  vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qcount, qprops.data());
  g_queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < qcount; ++i) {
    if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      g_queue_family = i;
      break;
    }
  }
  if (g_queue_family == UINT32_MAX) {
    vkDestroyInstance(g_instance, nullptr);
    g_instance = VK_NULL_HANDLE;
    return Error::make(ErrorCode::NotFound, "no Vulkan compute queue family");
  }
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = g_queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  err = vk_err(vkCreateDevice(g_phys, &dci, nullptr, &g_device), "CreateDevice");
  if (!err.ok()) {
    vkDestroyInstance(g_instance, nullptr);
    g_instance = VK_NULL_HANDLE;
    return err;
  }
  vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);
  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.queueFamilyIndex = g_queue_family;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  err = vk_err(vkCreateCommandPool(g_device, &pci, nullptr, &g_pool), "CreateCommandPool");
  if (!err.ok()) {
    vkDestroyDevice(g_device, nullptr);
    g_device = VK_NULL_HANDLE;
    vkDestroyInstance(g_instance, nullptr);
    g_instance = VK_NULL_HANDLE;
    return err;
  }
  err = build_add_pipeline();
  if (!err.ok()) {
    destroy_add_pipeline();
  }
  g_inited = true;
  return Error::success();
#endif
}
void vulkan_shutdown() noexcept {
#if defined(UAII_HAVE_VULKAN_HEADERS)
  std::lock_guard<std::mutex> lock(g_mu);
  destroy_add_pipeline();
  for (auto& kv : g_allocs) {
    if (kv.second.mapped) {
      vkUnmapMemory(g_device, kv.second.memory);
    }
    if (kv.second.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(g_device, kv.second.buffer, nullptr);
    }
    if (kv.second.memory != VK_NULL_HANDLE) {
      vkFreeMemory(g_device, kv.second.memory, nullptr);
    }
  }
  g_allocs.clear();
  if (g_pool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(g_device, g_pool, nullptr);
    g_pool = VK_NULL_HANDLE;
  }
  if (g_device != VK_NULL_HANDLE) {
    vkDestroyDevice(g_device, nullptr);
    g_device = VK_NULL_HANDLE;
  }
  if (g_instance != VK_NULL_HANDLE) {
    vkDestroyInstance(g_instance, nullptr);
    g_instance = VK_NULL_HANDLE;
  }
  g_inited = false;
#endif
}
Error vulkan_allocate(std::size_t bytes, void** out_ptr) {
#if !defined(UAII_HAVE_VULKAN_HEADERS)
  (void)bytes;
  (void)out_ptr;
  return vulkan_init();
#else
  if (out_ptr == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "vulkan_allocate out null");
  }
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_inited) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan not initialized");
  }
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = bytes == 0 ? 1 : bytes;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  AllocRec rec;
  Error err = vk_err(vkCreateBuffer(g_device, &bci, nullptr, &rec.buffer), "CreateBuffer");
  if (!err.ok()) {
    return err;
  }
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(g_device, rec.buffer, &req);
  const uint32_t mem_type =
      find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mem_type == UINT32_MAX) {
    vkDestroyBuffer(g_device, rec.buffer, nullptr);
    return Error::make(ErrorCode::Internal, "Vulkan host-visible memory type missing");
  }
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mem_type;
  err = vk_err(vkAllocateMemory(g_device, &mai, nullptr, &rec.memory), "AllocateMemory");
  if (!err.ok()) {
    vkDestroyBuffer(g_device, rec.buffer, nullptr);
    return err;
  }
  err = vk_err(vkBindBufferMemory(g_device, rec.buffer, rec.memory, 0), "BindBufferMemory");
  if (!err.ok()) {
    vkFreeMemory(g_device, rec.memory, nullptr);
    vkDestroyBuffer(g_device, rec.buffer, nullptr);
    return err;
  }
  err = vk_err(vkMapMemory(g_device, rec.memory, 0, req.size, 0, &rec.mapped), "MapMemory");
  if (!err.ok()) {
    vkFreeMemory(g_device, rec.memory, nullptr);
    vkDestroyBuffer(g_device, rec.buffer, nullptr);
    return err;
  }
  rec.bytes = bytes;
  g_allocs[rec.mapped] = rec;
  *out_ptr = rec.mapped;
  return Error::success();
#endif
}
Error vulkan_free(void* ptr) noexcept {
#if !defined(UAII_HAVE_VULKAN_HEADERS)
  (void)ptr;
  return Error::success();
#else
  if (ptr == nullptr) {
    return Error::success();
  }
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_allocs.find(ptr);
  if (it == g_allocs.end()) {
    return Error::make(ErrorCode::NotFound, "Vulkan pointer not owned");
  }
  if (it->second.mapped) {
    vkUnmapMemory(g_device, it->second.memory);
  }
  vkDestroyBuffer(g_device, it->second.buffer, nullptr);
  vkFreeMemory(g_device, it->second.memory, nullptr);
  g_allocs.erase(it);
  return Error::success();
#endif
}
Error vulkan_copy_h2d(const void* host, void* device, std::size_t bytes) {
  if (bytes == 0) return Error::success();
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "vulkan copy_h2d null");
  }
  std::memcpy(device, host, bytes);
  return Error::success();
}
Error vulkan_copy_d2h(const void* device, void* host, std::size_t bytes) {
  if (bytes == 0) return Error::success();
  if (host == nullptr || device == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "vulkan copy_d2h null");
  }
  std::memcpy(host, device, bytes);
  return Error::success();
}
Error vulkan_copy_d2d(const void* src, void* dst, std::size_t bytes) {
  if (bytes == 0) return Error::success();
  if (src == nullptr || dst == nullptr) {
    return Error::make(ErrorCode::InvalidArgument, "vulkan copy_d2d null");
  }
  std::memcpy(dst, src, bytes);
  return Error::success();
}
Error vulkan_synchronize() {
#if defined(UAII_HAVE_VULKAN_HEADERS)
  if (g_device != VK_NULL_HANDLE) {
    return vk_err(vkDeviceWaitIdle(g_device), "DeviceWaitIdle");
  }
#endif
  return Error::success();
}
Error vulkan_dispatch(const std::string& op_name, const std::string&,
                      const std::vector<kernels::TensorView>& inputs,
                      std::vector<kernels::TensorView>* outputs,
                      const std::vector<ir::Attribute>& attrs) {
#if !defined(UAII_HAVE_VULKAN_HEADERS)
  (void)op_name;
  (void)inputs;
  (void)outputs;
  (void)attrs;
  return Error::make(ErrorCode::NotImplemented, "Vulkan headers unavailable");
#else
  if (!g_inited) {
    return Error::make(ErrorCode::InvalidArgument, "Vulkan not initialized");
  }
  if (op_name == "Add") {
    Error err = dispatch_add_gpu(inputs, outputs);
    if (err.ok()) {
      return err;
    }
    return dispatch_add_mapped(inputs, outputs);
  }
  if (op_name == "MatMul" || op_name == "MatMulRelu") {
    Error err = dispatch_matmul_mapped(inputs, outputs, attrs);
    if (!err.ok() || op_name == "MatMul") {
      return err;
    }
    return Error::make(ErrorCode::NotImplemented, "Vulkan MatMulRelu not implemented");
  }
  if (op_name == "RMSNorm") {
    return dispatch_rmsnorm_mapped(inputs, outputs, attrs);
  }
  return Error::make(ErrorCode::NotImplemented,
                     "Vulkan native op not implemented: " + op_name);
#endif
}
}  // namespace native
}  // namespace backends
}  // namespace uaii
