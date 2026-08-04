uaii_add_module(storage
  SOURCES
    src/file_provider.cpp
    src/streaming.cpp
  DEPS uaii::core uaii::ir uaii::planner
)
