# Plugin Marketplace — Design Notes (Future)

**Status:** Design only (Phase 7). No marketplace service ships in-tree.

## Goal

Let third parties publish loaders, operators, backends, quantizers, and profilers that the UAII host discovers via the existing C plugin ABI (`UAII_PLUGIN_ABI_VERSION`).

## Principles

1. **ABI first** — packages must export `uaii_plugin_get_info` / `init` / `shutdown` with a matching ABI major.
2. **Capability manifests** — each listing declares kind, supported ops/dtypes, license, and minimum UAII C API version.
3. **Signed artifacts** — optional checksum + signature in the index; host verifies before `dlopen`.
4. **Sandboxed trust levels** — `trusted` (in-process) vs `isolated` (future out-of-process) for untrusted plugins.
5. **No central lock-in** — index format is JSON over HTTPS; multiple registries can coexist (`plugin.dirs` + remote URLs later).

## Proposed listing schema (sketch)

```json
{
  "name": "acme-cuda-flash",
  "kind": "backend",
  "version": "1.2.0",
  "uaii_plugin_abi": 1,
  "uaii_c_api_min": "1.0.0",
  "license": "Apache-2.0",
  "artifact": {
    "url": "https://example.com/acme-cuda-flash-1.2.0.zip",
    "sha256": "…"
  },
  "platforms": ["linux-x86_64", "windows-x86_64"]
}
```

## Host integration (later)

- `uaii plugin search|install|list` CLI
- Config: `plugin.registry_urls = ["https://…"]`
- Doctor reports installed marketplace plugins alongside local `plugin.dirs`

## Non-goals (now)

- Billing, accounts, or a hosted UAII backend for the documentation site
- Automatic binary builds for every plugin in CI of this repository
