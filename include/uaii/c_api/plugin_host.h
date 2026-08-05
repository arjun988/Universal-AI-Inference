#ifndef UAII_C_API_PLUGIN_HOST_H_
#define UAII_C_API_PLUGIN_HOST_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "uaii/c_api/plugin_abi.h"

#include <stdint.h>

/**
 * Optional host registration API passed to plugins that export
 * `uaii_plugin_register`. Enables OPERATOR plugins to inject kernels into
 * the CPU hot path (and future LOADER/BACKEND hooks).
 */
typedef int (*uaii_host_op_fn)(const void* /*inputs_blob*/,
                               void* /*outputs_blob*/,
                               const void* /*attrs_blob*/,
                               void* userdata);

typedef struct uaii_host_api {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Register a named operator. Returns 0 on success. */
  int (*register_op)(const char* op_name, uaii_host_op_fn fn, void* userdata);
  /** Register a loader probe callback (path → 1 if accepted). Optional. */
  int (*register_loader_probe)(const char* name,
                               int (*accepts)(const char* path, void* userdata),
                               void* userdata);
} uaii_host_api;

typedef int (*uaii_plugin_register_fn)(const uaii_host_api* host);

#ifdef __cplusplus
}
#endif

#endif /* UAII_C_API_PLUGIN_HOST_H_ */
