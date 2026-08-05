uaii_add_module(kernels
  SOURCES
    src/matmul.cpp
    src/gemm_ref.cpp
    src/gemm_onednn.cpp
    src/gemm_openblas.cpp
    src/softmax.cpp
    src/norm.cpp
    src/elementwise.cpp
    src/transformer.cpp
    src/quant_gemm.cpp
    src/dispatch.cpp
    src/thread_pool.cpp
    src/operator_host.cpp
    src/plugin_host_bridge.cpp
  DEPS uaii::core uaii::ir uaii::quant
)

find_package(Threads REQUIRED)
target_link_libraries(uaii_kernels PUBLIC Threads::Threads)

if(UAII_WITH_ONEDNN)
  find_package(DNNL CONFIG QUIET)
  if(DNNL_FOUND)
    target_compile_definitions(uaii_kernels PUBLIC UAII_HAVE_ONEDNN=1)
    target_link_libraries(uaii_kernels PUBLIC DNNL::dnnl)
  else()
    message(WARNING "UAII_WITH_ONEDNN=ON but DNNL not found; using ref GEMM")
  endif()
endif()

if(UAII_WITH_OPENBLAS)
  find_package(OpenBLAS QUIET)
  if(OpenBLAS_FOUND)
    target_compile_definitions(uaii_kernels PUBLIC UAII_HAVE_OPENBLAS=1)
    target_link_libraries(uaii_kernels PUBLIC ${OpenBLAS_LIBRARIES})
    target_include_directories(uaii_kernels PUBLIC ${OpenBLAS_INCLUDE_DIRS})
  else()
    find_library(OPENBLAS_LIB openblas)
    if(OPENBLAS_LIB)
      target_compile_definitions(uaii_kernels PUBLIC UAII_HAVE_OPENBLAS=1)
      target_link_libraries(uaii_kernels PUBLIC ${OPENBLAS_LIB})
    else()
      message(WARNING "UAII_WITH_OPENBLAS=ON but OpenBLAS not found; using ref GEMM")
    endif()
  endif()
endif()
