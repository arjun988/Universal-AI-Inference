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

add_executable(uaii_hf_loader_test unit/hf_loader_test.cpp)
target_link_libraries(uaii_hf_loader_test PRIVATE uaii::loaders uaii::ir uaii::core)
uaii_set_warnings(uaii_hf_loader_test)
add_test(NAME uaii_hf_loader_test COMMAND uaii_hf_loader_test)

add_executable(uaii_p0_rope_tokenizer_test unit/p0_rope_tokenizer_test.cpp)
target_link_libraries(uaii_p0_rope_tokenizer_test PRIVATE
  uaii::kernels uaii::tokenizers uaii::runtime uaii::loaders uaii::backends uaii::ir uaii::core)
uaii_set_warnings(uaii_p0_rope_tokenizer_test)
add_test(NAME uaii_p0_rope_tokenizer_test COMMAND uaii_p0_rope_tokenizer_test)
add_executable(uaii_storage_plan_test unit/storage_plan_test.cpp)
target_link_libraries(uaii_storage_plan_test PRIVATE uaii::planner uaii::ir uaii::core)
uaii_set_warnings(uaii_storage_plan_test)
add_test(NAME uaii_storage_plan_test COMMAND uaii_storage_plan_test)

add_executable(uaii_kv_cache_test unit/kv_cache_test.cpp)
target_link_libraries(uaii_kv_cache_test PRIVATE uaii::runtime uaii::core uaii::kernels)
uaii_set_warnings(uaii_kv_cache_test)
add_test(NAME uaii_kv_cache_test COMMAND uaii_kv_cache_test)

add_executable(uaii_memory_preset_test unit/memory_preset_test.cpp)
target_link_libraries(uaii_memory_preset_test PRIVATE uaii::runtime uaii::ir uaii::kernels uaii::core)
uaii_set_warnings(uaii_memory_preset_test)
add_test(NAME uaii_memory_preset_test COMMAND uaii_memory_preset_test)

# Ensure Windows finds uaii_capi.dll and the MinGW runtime. Semicolons inside
# PATH must be escaped or CTest treats them as separate environment entries.
string(REPLACE ";" "\\;" _uaii_env_path "$ENV{PATH}")
set_tests_properties(uaii_runtime_test PROPERTIES
  ENVIRONMENT "PATH=$<TARGET_FILE_DIR:uaii_capi>\\;${_uaii_env_path}")

# Demo / CLI smoke (skipped if binary path missing — CI sets WORKING_DIRECTORY)
add_test(NAME uaii_cli_inspect_help
  COMMAND $<TARGET_FILE:uaii> inspect --help)
set_tests_properties(uaii_cli_inspect_help PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  ENVIRONMENT "PATH=$<TARGET_FILE_DIR:uaii>\\;${_uaii_env_path}")
add_test(NAME uaii_cli_demo_toy_mlp
  COMMAND $<TARGET_FILE:uaii> run --demo toy_mlp)
set_tests_properties(uaii_cli_demo_toy_mlp PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  ENVIRONMENT "PATH=$<TARGET_FILE_DIR:uaii>\\;${_uaii_env_path}")
