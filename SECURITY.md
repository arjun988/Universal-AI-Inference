# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (development) | Yes — best effort |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Prefer:

1. GitHub **Private vulnerability reporting** on this repository (Security → Advisories)
2. Email: **security@uaii.dev** (placeholder — replace with maintainer address before public release)

Include:

- Description of the issue
- Impact assessment (if known)
- Steps to reproduce
- Affected commit / tag

We will acknowledge reports as quickly as possible and coordinate a fix and disclosure timeline.

## Trust model

- **Plugins** load **in-process** (`dlopen` / `LoadLibraryW`). Treat `plugin.dirs` like `PATH`: only trusted directories.
- **Weight / IR paths** may be sandboxed via `SessionOptions.weights_sandbox` / C API `weights_sandbox`.
- Untrusted model files should be treated as untrusted input (validate IR; prefer sandboxed weight dirs).
