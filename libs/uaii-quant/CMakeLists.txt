uaii_add_module(quant
  SOURCES
    src/codec.cpp
    src/gguf_dequant.cpp
    src/quantizer.cpp
  DEPS uaii::core uaii::ir
)
