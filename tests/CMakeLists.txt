# Phase 1 unit tests (GoogleTest can be wired later).
# Build/run instructions are documented in the root README — do not assume
# dependencies are installed in every environment.

add_executable(uaii_smoke_test
  unit/smoke_test.cpp
)

target_link_libraries(uaii_smoke_test PRIVATE uaii::core)
uaii_set_warnings(uaii_smoke_test)

add_test(NAME uaii_smoke_test COMMAND uaii_smoke_test)
