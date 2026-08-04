add_executable(uaii
  src/main.cpp
  src/commands/doctor.cpp
  src/commands/ir_commands.cpp
  src/commands/run.cpp
)

target_include_directories(uaii PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(uaii
  PRIVATE
    uaii::core
    uaii::ir
    uaii::memory
    uaii::storage
    uaii::planner
    uaii::kernels
    uaii::backends
    uaii::loaders
    uaii::runtime
    uaii::profiler
)

uaii_set_warnings(uaii)

install(TARGETS uaii RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
