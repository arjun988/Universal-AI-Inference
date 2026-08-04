uaii_add_module(runtime
  SOURCES
    src/scheduler_cpu.cpp
    src/session.cpp
    src/demos.cpp
  DEPS uaii::core uaii::ir uaii::memory uaii::storage uaii::planner uaii::kernels uaii::backends
)
