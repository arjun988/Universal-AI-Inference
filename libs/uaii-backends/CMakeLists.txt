uaii_add_module(backends
  SOURCES
    src/host_executable.cpp
    src/cpu_backend.cpp
    src/cuda_backend.cpp
    src/metal_backend.cpp
    src/vulkan_backend.cpp
    src/webgpu_backend.cpp
    src/rocm_backend.cpp
    src/factory.cpp
    src/parity.cpp
    src/native_unavailable.cpp
    src/native_scaffold.cpp
  DEPS uaii::core uaii::ir uaii::memory uaii::kernels
)

if(UAII_WITH_CUDA)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_CUDA=1)
endif()
if(UAII_WITH_METAL)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_METAL=1)
endif()
if(UAII_WITH_VULKAN)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_VULKAN=1)
endif()
if(UAII_WITH_WEBGPU)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_WEBGPU=1)
endif()
if(UAII_WITH_ROCM)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_ROCM=1)
endif()
