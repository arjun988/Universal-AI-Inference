#ifndef UAII_C_API_PLUGIN_ABI_H_
#define UAII_C_API_PLUGIN_ABI_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/** Increment when the C plugin ABI breaks. Host rejects mismatched majors. */
#define UAII_PLUGIN_ABI_VERSION 1

#define UAII_PLUGIN_NAME_MAX 64
#define UAII_PLUGIN_VERSION_MAX 32
#define UAII_PLUGIN_DESC_MAX 256

typedef enum uaii_plugin_kind {
  UAII_PLUGIN_KIND_UNKNOWN = 0,
  UAII_PLUGIN_KIND_LOADER = 1,
  UAII_PLUGIN_KIND_OPERATOR = 2,
  UAII_PLUGIN_KIND_BACKEND = 3,
  UAII_PLUGIN_KIND_STORAGE = 4,
  UAII_PLUGIN_KIND_SCHEDULER = 5,
  UAII_PLUGIN_KIND_TOKENIZER = 6,
  UAII_PLUGIN_KIND_QUANTIZATION = 7,
  UAII_PLUGIN_KIND_PROFILER = 8,
  UAII_PLUGIN_KIND_OPTIMIZER = 9,
  UAII_PLUGIN_KIND_PROBE = 10
} uaii_plugin_kind;

typedef struct uaii_plugin_info {
  uint32_t abi_version;
  uaii_plugin_kind kind;
  char name[UAII_PLUGIN_NAME_MAX];
  char version[UAII_PLUGIN_VERSION_MAX];
  char description[UAII_PLUGIN_DESC_MAX];
} uaii_plugin_info;

/**
 * Required entry points for every UAII plugin dynamic library.
 *
 *   uaii_plugin_get_info
 *   uaii_plugin_init
 *   uaii_plugin_shutdown
 */
typedef const uaii_plugin_info* (*uaii_plugin_get_info_fn)(void);
typedef int (*uaii_plugin_init_fn)(void);
typedef void (*uaii_plugin_shutdown_fn)(void);

#ifdef __cplusplus
}
#endif

#endif  /* UAII_C_API_PLUGIN_ABI_H_ */
