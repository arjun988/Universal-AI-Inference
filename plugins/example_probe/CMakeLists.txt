add_library(uaii_example_probe SHARED plugin.cpp)

target_include_directories(uaii_example_probe PRIVATE "${UAII_INCLUDE_DIR}")

# Plugins must export a stable C ABI; keep symbols visible.
set_target_properties(uaii_example_probe PROPERTIES
  OUTPUT_NAME "uaii_example_probe"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

uaii_set_warnings(uaii_example_probe)
add_dependencies(uaii_example_probe uaii)

# Copy next to CLI for convenient discovery during local builds.
add_custom_command(TARGET uaii_example_probe POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:uaii>/plugins"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:uaii_example_probe>"
          "$<TARGET_FILE_DIR:uaii>/plugins/"
  COMMENT "Installing example plugin next to uaii CLI"
)

install(TARGETS uaii_example_probe
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/uaii/plugins
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/plugins
)
