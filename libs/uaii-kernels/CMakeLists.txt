uaii_add_module(kernels
  SOURCES
    src/matmul.cpp
    src/softmax.cpp
    src/norm.cpp
    src/elementwise.cpp
    src/transformer.cpp
    src/dispatch.cpp
  DEPS uaii::core uaii::ir
)
