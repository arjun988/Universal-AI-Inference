add_library(uaii_example_op SHARED plugin.cpp)

target_include_directories(uaii_example_op PRIVATE "${UAII_INCLUDE_DIR}")
target_link_libraries(uaii_example_op PRIVATE uaii::kernels uaii::ir uaii::core)

set_target_properties(uaii_example_op PROPERTIES
  OUTPUT_NAME "uaii_example_op"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

uaii_set_warnings(uaii_example_op)
add_dependencies(uaii_example_op uaii)

add_custom_command(TARGET uaii_example_op POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:uaii>/plugins"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:uaii_example_op>"
          "$<TARGET_FILE_DIR:uaii>/plugins/"
  COMMENT "Installing example_op plugin next to uaii CLI"
)

install(TARGETS uaii_example_op
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/uaii/plugins
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/plugins
)
