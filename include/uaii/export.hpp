#pragma once

// Shared/static library export macros for future shared builds.
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(UAII_BUILD_SHARED)
#    if defined(UAII_EXPORTS)
#      define UAII_API __declspec(dllexport)
#    else
#      define UAII_API __declspec(dllimport)
#    endif
#  else
#    define UAII_API
#  endif
#else
#  if defined(UAII_BUILD_SHARED)
#    define UAII_API __attribute__((visibility("default")))
#  else
#    define UAII_API
#  endif
#endif
