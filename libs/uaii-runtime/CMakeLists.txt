uaii_add_module(runtime
  SOURCES
    src/scheduler_cpu.cpp
    src/session.cpp
    src/demos.cpp
    src/demos_phase4.cpp
    src/demos_phase5.cpp
  DEPS uaii::core uaii::ir uaii::memory uaii::storage uaii::planner uaii::kernels uaii::backends uaii::loaders uaii::tokenizers
)
