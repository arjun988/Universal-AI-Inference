# Contributing to Universal AI Inference Runtime

Thanks for your interest in contributing to **UAII Runtime**.

## Ground rules

1. Core must not import model-specific code.
2. External formats convert to **UAII IR** before execution (Phase 2+).
3. Backends understand only UAII IR + planned kernels.
4. Prefer new plugins over core changes.
5. Public docs use **Universal AI Inference Runtime** / **UAII Runtime** only.

## Development setup

See the root [README.md](./README.md) for prerequisites and build steps.

Recommended before opening a PR:

1. Configure and build with CMake.
2. Run `uaii doctor` (and optionally `--load-plugins`).
3. Run the smoke test target if you changed `uaii-core`.
4. Format C++ with `clang-format` (style file is `.clang-format`).

> Note: CI and local verification commands are documented in the README. Follow those instructions in your own environment; this repository does not assume tools are pre-installed.

## Project layout

| Path | Role |
|---|---|
| `include/uaii/` | Public headers / interfaces |
| `libs/uaii-core/` | Errors, logging, config, plugins |
| `libs/uaii-*/` | Modular libraries (many Phase 1 stubs) |
| `plugins/` | Example / reference plugins |
| `docs/` | Vision, plan, architecture, features |
| `tests/` | Unit / smoke tests |

## Coding standards

- **Language:** C++17
- **Style:** Google-based via `.clang-format`
- **Errors:** Return `uaii::Error`; avoid exceptions on hot paths
- **Logging:** Use `uaii::log::*` with a short component name
- **ABI:** Plugin entry points stay `extern "C"` and versioned

## Pull requests

- Keep PRs focused and modular.
- Update docs when behavior or public APIs change.
- Add or extend tests for core behavior changes.
- Describe *why* the change is needed in the PR body.

## Reporting issues

Use GitHub Issues. Include OS, compiler, CMake version, and the exact command that failed.
