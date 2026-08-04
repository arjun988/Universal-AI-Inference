uaii_add_module(planner
  SOURCES
    src/fusion.cpp
    src/memory_plan.cpp
    src/storage_plan.cpp
    src/cache.cpp
    src/optimize.cpp
  DEPS uaii::core uaii::ir
)
