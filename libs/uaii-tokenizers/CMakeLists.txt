uaii_add_module(tokenizers
  SOURCES
    src/simple_tokenizer.cpp
    src/bpe_tokenizer.cpp
    src/sentencepiece_tokenizer.cpp
    src/gguf_tokenizer.cpp
  DEPS uaii::core uaii::loaders
)

if(UAII_WITH_SENTENCEPIECE)
  find_package(SentencePiece QUIET)
  if(SentencePiece_FOUND)
    target_compile_definitions(uaii_tokenizers PUBLIC UAII_HAVE_SENTENCEPIECE=1)
    target_link_libraries(uaii_tokenizers PUBLIC sentencepiece)
  else()
    find_library(SP_LIB sentencepiece)
    if(SP_LIB)
      target_compile_definitions(uaii_tokenizers PUBLIC UAII_HAVE_SENTENCEPIECE=1)
      target_link_libraries(uaii_tokenizers PUBLIC ${SP_LIB})
    else()
      message(WARNING "UAII_WITH_SENTENCEPIECE=ON but library not found")
    endif()
  endif()
endif()
