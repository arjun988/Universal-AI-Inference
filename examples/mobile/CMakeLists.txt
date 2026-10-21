cmake_minimum_required(VERSION 3.20)

project(uaii_phone_sample LANGUAGES C CXX)

set(UAII_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(UAII_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(UAII_BUILD_PLUGINS OFF CACHE BOOL "" FORCE)
set(UAII_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(UAII_WITH_CUDA OFF CACHE BOOL "" FORCE)
set(UAII_BUILD_MOBILE_SAMPLE OFF CACHE BOOL "" FORCE)

get_filename_component(UAII_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_subdirectory("${UAII_ROOT}" "${CMAKE_BINARY_DIR}/uaii")

add_executable(uaii_phone "${CMAKE_CURRENT_LIST_DIR}/main.c")
set_target_properties(uaii_phone PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(uaii_phone PRIVATE uaii::capi)
