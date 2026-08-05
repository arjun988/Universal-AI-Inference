uaii_add_module(kernels
  SOURCES
    src/matmul.cpp
    src/softmax.cpp
    src/norm.cpp
    src/elementwise.cpp
    src/transformer.cpp
    src/dispatch.cpp
    src/thread_pool.cpp
    src/operator_host.cpp
    src/plugin_host_bridge.cpp
  DEPS uaii::core uaii::ir
)

# Threads for parallel MatMul
find_package(Threads REQUIRED)
target_link_libraries(uaii_kernels PUBLIC Threads::Threads)
