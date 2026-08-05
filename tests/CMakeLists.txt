# Unit / integration tests (no external GoogleTest required).

add_executable(uaii_smoke_test unit/smoke_test.cpp)
target_link_libraries(uaii_smoke_test PRIVATE uaii::core)
uaii_set_warnings(uaii_smoke_test)
add_test(NAME uaii_smoke_test COMMAND uaii_smoke_test)

add_executable(uaii_runtime_test unit/runtime_test.cpp)
target_link_libraries(uaii_runtime_test PRIVATE
  uaii::core uaii::ir uaii::kernels uaii::runtime uaii::capi uaii::backends)
uaii_set_warnings(uaii_runtime_test)
add_test(NAME uaii_runtime_test COMMAND uaii_runtime_test)

add_executable(uaii_quant_dequant_test unit/quant_dequant_test.cpp)
target_link_libraries(uaii_quant_dequant_test PRIVATE uaii::core uaii::quant)
uaii_set_warnings(uaii_quant_dequant_test)
add_test(NAME uaii_quant_dequant_test COMMAND uaii_quant_dequant_test)

# Ensure Windows finds uaii_capi.dll (and deps) next to / via PATH.
set(_uaii_test_path "$<TARGET_FILE_DIR:uaii_capi>;$ENV{PATH}")
set_tests_properties(uaii_runtime_test PROPERTIES
  ENVIRONMENT "PATH=${_uaii_test_path}")

# Demo / CLI smoke (skipped if binary path missing — CI sets WORKING_DIRECTORY)
add_test(NAME uaii_cli_demo_toy_mlp
  COMMAND $<TARGET_FILE:uaii> run --demo toy_mlp)
set_tests_properties(uaii_cli_demo_toy_mlp PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  ENVIRONMENT "PATH=$<TARGET_FILE_DIR:uaii>;$ENV{PATH}")
