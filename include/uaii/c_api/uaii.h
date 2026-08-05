#ifndef UAII_C_API_UAII_H_
#define UAII_C_API_UAII_H_

/**
 * Universal AI Inference Runtime — stable C ABI (Phase 7).
 *
 * Guarantees: semantic versioning of this header + shared library surface.
 * See docs/c_api_stability.md and include/uaii/c_api/version.h.
 */

#include "uaii/c_api/version.h"
#include "uaii/c_api/plugin_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(UAII_CAPI_BUILD_SHARED)
#    if defined(UAII_CAPI_EXPORTS)
#      define UAII_CAPI_API __declspec(dllexport)
#    else
#      define UAII_CAPI_API __declspec(dllimport)
#    endif
#  else
#    define UAII_CAPI_API
#  endif
#else
#  if defined(UAII_CAPI_BUILD_SHARED)
#    define UAII_CAPI_API __attribute__((visibility("default")))
#  else
#    define UAII_CAPI_API
#  endif
#endif

typedef enum uaii_status {
  UAII_OK = 0,
  UAII_ERR_INVALID_ARGUMENT = 1,
  UAII_ERR_NOT_FOUND = 2,
  UAII_ERR_IO = 3,
  UAII_ERR_CONFIG = 4,
  UAII_ERR_PLUGIN = 5,
  UAII_ERR_ABI = 6,
  UAII_ERR_NOT_IMPLEMENTED = 7,
  UAII_ERR_INTERNAL = 8
} uaii_status;

typedef enum uaii_weight_init {
  UAII_WEIGHT_INIT_NONE = 0,
  UAII_WEIGHT_INIT_ZEROS = 1,
  UAII_WEIGHT_INIT_ONES = 2,
  UAII_WEIGHT_INIT_SEQUENCE = 3
} uaii_weight_init;

typedef struct uaii_session uaii_session;

typedef struct uaii_session_options {
  /** Must be set to sizeof(uaii_session_options) by callers (ABI-safe growth). */
  uint32_t struct_size;
  /** Backend name: "cpu", "cuda", … (default "cpu" if NULL) */
  const char* backend;
  /** Directory for weight_ref resolution (may be NULL) */
  const char* weights_dir;
  /** Default: UAII_WEIGHT_INIT_NONE (fail closed if weights missing). */
  uaii_weight_init weight_init;
  int enable_fusion;       /* non-zero = on (default 1) */
  int enable_memory_reuse; /* non-zero = on (default 1) */
  int enable_profiler;     /* non-zero = on */
  /** If non-NULL and profiling on, write chrome-trace JSON after run */
  const char* profile_trace_path;
  /** Memory budget in bytes; 0 = unlimited */
  uint64_t budget_bytes;
  int enable_streaming; /* non-zero enables streaming under budget */
  /** If non-zero, allow synthetic weight_init when weight_ref load fails. */
  int allow_missing_weights;
  /** Optional sandbox root; weight paths must stay under this directory. */
  const char* weights_sandbox;
} uaii_session_options;

/** Fill options with safe defaults. */
UAII_CAPI_API void uaii_session_options_init(uaii_session_options* opts);

/** Library package version (CMake project version). */
UAII_CAPI_API uaii_status uaii_get_version(int* major, int* minor, int* patch);
UAII_CAPI_API const char* uaii_get_version_string(void);

/** Stable C API semver (independent of package patch bumps when ABI unchanged). */
UAII_CAPI_API void uaii_get_c_api_version(int* major, int* minor, int* patch);
UAII_CAPI_API const char* uaii_get_c_api_version_string(void);

UAII_CAPI_API const char* uaii_status_name(uaii_status status);
/** Thread-local last error message (valid until next C API call on this thread). */
UAII_CAPI_API const char* uaii_last_error(void);

/**
 * Create a session from a UAII IR file (.uaii.json / .uaii) or a model file
 * (GGUF / Safetensors) — format is detected by extension / loader registry.
 */
UAII_CAPI_API uaii_status uaii_session_create(const char* path,
                                              const uaii_session_options* opts,
                                              uaii_session** out);

UAII_CAPI_API void uaii_session_destroy(uaii_session* session);

UAII_CAPI_API uaii_status uaii_session_set_f32(uaii_session* session,
                                               const char* tensor_name,
                                               const float* data,
                                               size_t n);

UAII_CAPI_API uaii_status uaii_session_get_f32(uaii_session* session,
                                               const char* tensor_name,
                                               float* out,
                                               size_t capacity,
                                               size_t* out_n);

UAII_CAPI_API uaii_status uaii_session_run(uaii_session* session);

/** Copy profiler summary into buf (NUL-terminated). */
UAII_CAPI_API uaii_status uaii_session_profile_summary(uaii_session* session,
                                                       char* buf,
                                                       size_t capacity);

/** Write chrome-trace JSON for the session profiler. */
UAII_CAPI_API uaii_status uaii_session_write_trace(uaii_session* session,
                                                   const char* path);

/** Convert GGUF/Safetensors → UAII IR path. */
UAII_CAPI_API uaii_status uaii_convert_model(const char* input_path,
                                             const char* output_path);

#ifdef __cplusplus
}
#endif

#endif /* UAII_C_API_UAII_H_ */
