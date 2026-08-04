# UAII C API Stability Guarantees

**Surface:** [`include/uaii/c_api/uaii.h`](../include/uaii/c_api/uaii.h)  
**API version macros:** [`include/uaii/c_api/version.h`](../include/uaii/c_api/version.h)  
**Current C API:** `UAII_C_API_VERSION` = **1.0.0**

## Semantic versioning

| Bump | When |
|---|---|
| **MAJOR** | Removed/renamed symbols, changed struct layouts, changed semantics of existing functions |
| **MINOR** | New functions/fields (callers compiled against older minors keep working) |
| **PATCH** | Bug fixes that preserve documented behavior |

Package version (`uaii_get_version` / CMake `PROJECT_VERSION`) may advance independently when only C++ internals change. Bindings should gate on **`uaii_get_c_api_version`**.

## Compatibility rules (v1)

1. Opaque handles (`uaii_session*`) — never dereference fields from client code.
2. All functions return `uaii_status`; details via `uaii_last_error()`.
3. String arguments are UTF-8 paths / names; ownership stays with the caller.
4. `uaii_session_options` may grow at the end in minor releases; always call `uaii_session_options_init` before setting fields.
5. Plugin ABI (`UAII_PLUGIN_ABI_VERSION`) is **separate** from the C API version.

## Shared library

The CMake target `uaii_capi` builds the ABI-stable shared library (`uaii_capi.dll` / `libuaii_capi.so` / `libuaii_capi.dylib`).

## Deprecation

Symbols may be marked deprecated in a MINOR release and removed only in the next MAJOR.
