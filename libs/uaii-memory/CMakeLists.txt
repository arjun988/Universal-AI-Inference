uaii_add_module(memory
  SOURCES
    src/budget.cpp
    src/arena.cpp
    src/pool.cpp
    src/allocator.cpp
  DEPS uaii::core uaii::ir
)
