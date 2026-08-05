cmake_minimum_required(VERSION 3.20)

project(uaii
  VERSION 0.1.0
  DESCRIPTION "Universal AI Inference Runtime"
  HOMEPAGE_URL "https://github.com/arjunshukla/Universal-AI-Inference"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(UAIIOptions)
include(CompilerWarnings)
include(UAIIModule)
include(GNUInstallDirs)

# Public headers live in include/
set(UAII_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Generated version header
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uaii_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/uaii/version_generated.hpp"
  @ONLY
)

add_library(uaii_headers INTERFACE)
add_library(uaii::headers ALIAS uaii_headers)
target_include_directories(uaii_headers INTERFACE
  "$<BUILD_INTERFACE:${UAII_INCLUDE_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
install(TARGETS uaii_headers EXPORT uaiiTargets)

# --- Modular libraries -------------------------------------------------------
add_subdirectory(libs/uaii-core)
add_subdirectory(libs/uaii-ir)
add_subdirectory(libs/uaii-memory)
add_subdirectory(libs/uaii-planner)
add_subdirectory(libs/uaii-storage)
add_subdirectory(libs/uaii-kernels)
add_subdirectory(libs/uaii-backends)
add_subdirectory(libs/uaii-tokenizers)
add_subdirectory(libs/uaii-loaders)
add_subdirectory(libs/uaii-quant)
add_subdirectory(libs/uaii-profiler)
add_subdirectory(libs/uaii-runtime)
add_subdirectory(libs/uaii-capi)
add_subdirectory(libs/uaii-cli)

if(UAII_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

if(UAII_BUILD_PLUGINS)
  add_subdirectory(plugins/example_probe)
  add_subdirectory(plugins/example_op)
endif()

if(UAII_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# --- Install / find_package(uaii) --------------------------------------------
include(CMakePackageConfigHelpers)

install(DIRECTORY "${UAII_INCLUDE_DIR}/uaii"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/uaii/version_generated.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/uaii"
)

# Targets are installed with EXPORT via cmake/UAIIModule.cmake + uaii_core/capi.
install(EXPORT uaiiTargets
  FILE uaiiTargets.cmake
  NAMESPACE uaii::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/uaii
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uaiiConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/uaiiConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/uaii
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/uaiiConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/uaiiConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/uaiiConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/uaii
)

message(STATUS "UAII Runtime ${PROJECT_VERSION}")
message(STATUS "  Build shared libs : ${BUILD_SHARED_LIBS}")
message(STATUS "  Build tests       : ${UAII_BUILD_TESTS}")
message(STATUS "  Build plugins     : ${UAII_BUILD_PLUGINS}")
message(STATUS "  Build Python      : ${UAII_BUILD_PYTHON}")
message(STATUS "  Warnings as errors: ${UAII_WARNINGS_AS_ERRORS}")
