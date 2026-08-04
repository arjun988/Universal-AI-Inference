uaii_add_module(loaders
  SOURCES
    src/gguf.cpp
    src/safetensors.cpp
    src/registry.cpp
  DEPS uaii::core uaii::ir
)
