uaii_add_module(runtime
  SOURCES
    src/scheduler_cpu.cpp
    src/scheduler_device.cpp
    src/session.cpp
    src/sampling.cpp
    src/kv_cache.cpp
    src/demos.cpp
    src/demos_phase4.cpp
    src/demos_phase5.cpp
    src/demos_phase6.cpp
  DEPS uaii::core uaii::ir uaii::memory uaii::storage uaii::planner uaii::kernels uaii::backends uaii::loaders uaii::tokenizers uaii::quant uaii::profiler
)
