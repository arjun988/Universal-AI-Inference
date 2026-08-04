# Phase 1 — Foundation (Complete in-tree)

**Status:** Implemented in repository (verify locally via README build steps)

## Delivered

- CMake modular workspace (`libs/uaii-*`)
- Public headers under `include/uaii`
- `uaii-core`: errors, logging, config, plugin host
- C plugin ABI (`plugin_abi.h`) + example probe plugin
- Core interfaces: backend, loader, operator, storage, scheduler, tokenizer
- Stub libraries for later phases (IR, runtime, memory, …)
- CLI: `uaii doctor`, `version`, `help`
- OSS metadata: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, NOTICE, CI
- Product docs: vision, plan, architecture, feature (C++ tech stack)

## Verify locally

Follow root README — configure, build, run doctor, run smoke tests.
Do not assume the environment already has toolchains installed.
