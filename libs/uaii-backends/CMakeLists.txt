uaii_add_module(backends
  SOURCES
    src/cpu_backend.cpp
  DEPS uaii::core uaii::ir uaii::memory uaii::kernels
)
