add_library(uaii_core
  src/version.cpp
  src/error.cpp
  src/log.cpp
  src/config.cpp
  src/plugin.cpp
)

add_library(uaii::core ALIAS uaii_core)

target_link_libraries(uaii_core PUBLIC uaii::headers)

if(UNIX AND NOT APPLE)
  target_link_libraries(uaii_core PUBLIC dl)
endif()

# libstdc++fs required for std::filesystem on GCC < 9.1
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  target_link_libraries(uaii_core PUBLIC stdc++fs)
endif()

uaii_set_warnings(uaii_core)

set_target_properties(uaii_core PROPERTIES
  OUTPUT_NAME "uaii_core"
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS uaii_core
  EXPORT uaiiTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
