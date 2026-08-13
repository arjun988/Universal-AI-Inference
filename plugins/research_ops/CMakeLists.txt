add_library(uaii_research_ops SHARED plugin.cpp)

target_include_directories(uaii_research_ops PRIVATE "${UAII_INCLUDE_DIR}")
target_link_libraries(uaii_research_ops PRIVATE uaii::kernels uaii::ir uaii::core)

set_target_properties(uaii_research_ops PROPERTIES
  OUTPUT_NAME "uaii_research_ops"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

uaii_set_warnings(uaii_research_ops)
add_dependencies(uaii_research_ops uaii)

add_custom_command(TARGET uaii_research_ops POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:uaii>/plugins"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:uaii_research_ops>"
          "$<TARGET_FILE_DIR:uaii>/plugins/"
  COMMENT "Installing research_ops plugin next to uaii CLI"
)

install(TARGETS uaii_research_ops
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/uaii/plugins
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/plugins
)
