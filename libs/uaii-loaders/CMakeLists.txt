uaii_add_module(loaders
  SOURCES
    src/gguf.cpp
    src/safetensors.cpp
    src/onnx.cpp
    src/pytorch.cpp
    src/mlx.cpp
    src/registry.cpp
  DEPS uaii::core uaii::ir
)

# libstdc++fs for directory_iterator on older GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  target_link_libraries(uaii_loaders PUBLIC stdc++fs)
endif()
