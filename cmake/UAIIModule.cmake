# Declare a UAII library module.
# Usage: uaii_add_module(<short_name> [STUB] SOURCES ... [DEPS ...])
# Creates target uaii_<short_name> and alias uaii::<short_name>.
function(uaii_add_module short_name)
  cmake_parse_arguments(ARG "STUB" "" "SOURCES;DEPS" ${ARGN})

  set(target_name "uaii_${short_name}")

  add_library(${target_name} ${ARG_SOURCES})
  add_library(uaii::${short_name} ALIAS ${target_name})

  target_link_libraries(${target_name} PUBLIC uaii::headers ${ARG_DEPS})
  uaii_set_warnings(${target_name})

  set_target_properties(${target_name} PROPERTIES
    OUTPUT_NAME "${target_name}"
    POSITION_INDEPENDENT_CODE ON
  )

  if(ARG_STUB)
    target_compile_definitions(${target_name} PRIVATE UAII_MODULE_STUB=1)
  endif()

  install(TARGETS ${target_name}
    EXPORT uaiiTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endfunction()
