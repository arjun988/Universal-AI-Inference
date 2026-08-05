# Stable C ABI shared library (Phase 7). Always SHARED so language bindings can load it.
add_library(uaii_capi SHARED src/capi.cpp)
add_library(uaii::capi ALIAS uaii_capi)

target_link_libraries(uaii_capi
  PRIVATE
    uaii::core
    uaii::ir
    uaii::loaders
    uaii::runtime
    uaii::profiler
)

target_include_directories(uaii_capi
  PUBLIC
    "$<BUILD_INTERFACE:${UAII_INCLUDE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_definitions(uaii_capi
  PRIVATE UAII_CAPI_EXPORTS
  PUBLIC UAII_CAPI_BUILD_SHARED=1
)

uaii_set_warnings(uaii_capi)

set_target_properties(uaii_capi PROPERTIES
  OUTPUT_NAME "uaii_capi"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

install(TARGETS uaii_capi
  EXPORT uaiiTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
