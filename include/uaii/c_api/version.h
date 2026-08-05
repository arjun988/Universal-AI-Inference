#ifndef UAII_C_API_VERSION_H_
#define UAII_C_API_VERSION_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stable public C API semantic version (Phase 7).
 *
 * Policy (see docs/c_api_stability.md):
 *   - MAJOR: breaking changes to signatures / semantics
 *   - MINOR: backward-compatible additions
 *   - PATCH: bug fixes / docs only
 *
 * Plugin ABI (`UAII_PLUGIN_ABI_VERSION`) is versioned independently.
 */
#define UAII_C_API_VERSION_MAJOR 0
#define UAII_C_API_VERSION_MINOR 3
#define UAII_C_API_VERSION_PATCH 0

#define UAII_C_API_VERSION_STRING "0.3.0"

#ifdef __cplusplus
}
#endif

#endif /* UAII_C_API_VERSION_H_ */
