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
  DEPS uaii::core uaii::ir uaii::memory uaii::kernels
)

# --- CUDA (first-class device path) ------------------------------------------
if(UAII_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Required by CMake 3.18+ when enabling CUDA (empty arch list fails generate).
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      # Turing→Ada defaults. For RTX 50 / Blackwell (sm_120) use CUDA toolkit
      # >= 12.8 and pass -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120".
      set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING
          "CUDA architectures for UAII")
    endif()
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "UAII_WITH_CUDA=ON but no CUDA compiler found")
  endif()

  find_package(CUDAToolkit REQUIRED)

  target_sources(uaii_backends PRIVATE
    src/cuda_runtime_api.cpp
    src/cuda_kernels.cu
  )
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_CUDA=1)
  target_link_libraries(uaii_backends PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
  set_target_properties(uaii_backends PROPERTIES
    # Keep device code self-contained in the static lib so final link of `uaii`
    # does not need a separate CUDA device-link step.
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
  # Allow .cpp to include cuda_runtime.h / cublas_v2.h
  target_include_directories(uaii_backends PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# --- Metal -------------------------------------------------------------------
if(UAII_WITH_METAL)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_METAL=1)
  if(APPLE)
    target_sources(uaii_backends PRIVATE src/metal_runtime_api.mm)
    target_link_libraries(uaii_backends PUBLIC
      "-framework Metal"
      "-framework Foundation"
    )
  else()
    target_sources(uaii_backends PRIVATE src/metal_runtime_api.cpp)
  endif()
endif()

# --- Vulkan ------------------------------------------------------------------
if(UAII_WITH_VULKAN)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_VULKAN=1)
  target_sources(uaii_backends PRIVATE src/vulkan_runtime_api.cpp)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    target_link_libraries(uaii_backends PUBLIC Vulkan::Vulkan)
  else()
    message(STATUS "UAII_WITH_VULKAN=ON but Vulkan package not found; "
                   "vulkan_runtime_api.cpp will report missing headers at runtime")
  endif()
endif()

# --- WebGPU ------------------------------------------------------------------
if(UAII_WITH_WEBGPU)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_WEBGPU=1)
  target_sources(uaii_backends PRIVATE src/webgpu_runtime_api.cpp)
endif()

# --- ROCm / HIP --------------------------------------------------------------
if(UAII_WITH_ROCM)
  target_compile_definitions(uaii_backends PUBLIC UAII_WITH_ROCM=1)
  target_sources(uaii_backends PRIVATE src/rocm_runtime_api.cpp)
  find_package(hip QUIET)
  if(hip_FOUND)
    target_sources(uaii_backends PRIVATE src/rocm_kernels.hip)
    target_link_libraries(uaii_backends PUBLIC hip::host)
    set_source_files_properties(src/rocm_kernels.hip PROPERTIES LANGUAGE HIP)
  endif()
  find_package(rocblas QUIET)
  if(rocblas_FOUND)
    target_link_libraries(uaii_backends PUBLIC roc::rocblas)
  endif()
endif()
