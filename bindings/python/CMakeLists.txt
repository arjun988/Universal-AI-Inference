# Optional pybind11 extension. Requires Python3 + network/FetchContent on first configure.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_uaii MODULE src/module.cpp)
target_link_libraries(_uaii PRIVATE uaii::capi uaii::runtime uaii::loaders uaii::profiler)
target_include_directories(_uaii PRIVATE "${UAII_INCLUDE_DIR}")

# Install next to the pure-Python package when building with Python enabled.
set(_uaii_py_out "${CMAKE_SOURCE_DIR}/bindings/python/uaii")
set_target_properties(_uaii PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_uaii_py_out}"
  RUNTIME_OUTPUT_DIRECTORY "${_uaii_py_out}"
)

message(STATUS "UAII Python extension (_uaii) enabled")
