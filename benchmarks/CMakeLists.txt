# Microbenchmark harness (not part of ctest by default).

add_executable(uaii_bench src/uaii_bench.cpp)
target_link_libraries(uaii_bench PRIVATE
  uaii::core
  uaii::ir
  uaii::kernels
  uaii::quant
  uaii::runtime
  uaii::backends
)
uaii_set_warnings(uaii_bench)

install(TARGETS uaii_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
