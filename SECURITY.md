# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (development) | Yes — best effort |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Prefer one of:

1. GitHub **Private vulnerability reporting** on this repository (if enabled)
2. Email the maintainer listed in the repository profile / commit history

Include:

- Description of the issue
- Impact assessment (if known)
- Steps to reproduce
- Affected commit / tag

We will acknowledge reports as quickly as possible and coordinate a fix and disclosure timeline.

## Plugin trust model (Phase 1)

UAII plugins are loaded **in-process** as native dynamic libraries. Treat plugin
directories like `PATH`: only load plugins from trusted sources.
