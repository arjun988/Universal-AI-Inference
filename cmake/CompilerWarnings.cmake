function(uaii_set_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    if(UAII_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target_name} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wconversion -Wshadow -Wnon-virtual-dtor
      -Wold-style-cast -Wcast-align -Wunused
      -Woverloaded-virtual -Wnull-dereference
    )
    if(UAII_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()
  endif()
endfunction()
