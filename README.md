# Universal AI Inference Runtime (UAII Runtime)

**Universal AI Inference Runtime** is a modular execution platform for AI inference:
any model → **UAII IR** → any hardware.

It is designed as an *execution operating system* for inference—not a single-model
engine. Models, operators, backends, schedulers, and storage providers are plugins.

> Status: **Phase 1 — Foundation** (scaffolding, core services, interfaces, CLI doctor)

## Docs

| Document | Description |
|---|---|
| [docs/vision.md](docs/vision.md) | Mission and principles |
| [docs/plan.md](docs/plan.md) | Roadmap and **tech stack** |
| [docs/architecture.md](docs/architecture.md) | Module and IR architecture |
| [docs/feature.md](docs/feature.md) | Feature catalog |

## Tech stack (locked for this project)

| Layer | Choice |
|---|---|
| Core runtime | **C++17** |
| Build | **CMake 3.20+** |
| Plugins / SDK ABI | **C ABI** (`UAII_PLUGIN_ABI_VERSION`) |
| Config | TOML subset + `UAII_*` env overlays |
| Python SDK | Planned (Phase 7, pybind11/nanobind) — not required for core |

Rust is **not** used in the core.

## Repository layout

```
include/uaii/          Public headers (core + interfaces + C ABI)
libs/uaii-core/        Errors, logging, config, plugin host
libs/uaii-ir/          IR (Phase 2 stub)
libs/uaii-runtime/     Execution engine (Phase 3 stub)
libs/uaii-memory/      Allocators (Phase 3 stub)
libs/uaii-storage/     Storage engine (Phase 3/6 stub)
libs/uaii-planner/     Planner (Phase 3/6 stub)
libs/uaii-kernels/     Kernels (Phase 3 stub)
libs/uaii-backends/    Backends (Phase 3/5 stub)
libs/uaii-loaders/     Model loaders (Phase 4 stub)
libs/uaii-profiler/    Profiler (Phase 6 stub)
libs/uaii-cli/         `uaii` CLI
plugins/example_probe/ Example plugin for discovery/load validation
configs/uaii.toml      Default configuration
docs/                  Product docs
tests/                 Smoke / unit tests
```

## Prerequisites

Install these on your machine before building (this README documents the steps;
nothing is auto-installed by the project):

- A C++17 compiler (MSVC 2019+, Clang 10+, or GCC 9+)
- CMake 3.20 or newer
- Ninja or your platform’s default generator (optional but recommended)
- Git

Optional later: CUDA / Vulkan / Metal SDKs (Phase 5), Python 3.10+ (Phase 7).

## Build (run these yourself)

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel
```

On Windows (Visual Studio generator), prefer:

```bash
cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel
```

### Useful CMake options

| Option | Default | Meaning |
|---|---|---|
| `UAII_BUILD_TESTS` | `ON` | Build smoke tests |
| `UAII_BUILD_PLUGINS` | `ON` | Build example plugins |
| `UAII_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` / `/WX` |
| `UAII_WITH_CUDA` | `OFF` | Reserved (Phase 5) |

## Run CLI (after you build)

Executable name: **`uaii`**

```bash
# Help / version
./build/uaii help
./build/uaii version

# Environment + module + plugin diagnostics
./build/uaii doctor
./build/uaii doctor --load-plugins

# With config / log level
./build/uaii --config configs/uaii.toml --log-level debug doctor --load-plugins
```

On Windows multi-config builds the binary may be under `build/Release/uaii.exe`
(or `build/Debug/uaii.exe`). The example plugin is copied to a `plugins/`
directory next to the CLI when plugins are built.

### Phase 1 CLI commands

| Command | Status |
|---|---|
| `uaii doctor` | Implemented |
| `uaii version` / `uaii help` | Implemented |
| `uaii run` / `validate` / `convert` / … | Later phases |

## Tests (run these yourself — not executed by the authoring agent)

```bash
ctest --test-dir build --output-on-failure
# or, multi-config:
ctest --test-dir build -C Release --output-on-failure

# Direct smoke test binary (path may vary by generator):
./build/uaii_smoke_test
```

The smoke test covers version, `Error`, TOML-subset config parsing, and log level parsing.

## Configuration

Default file: [`configs/uaii.toml`](configs/uaii.toml)

Environment overlays (examples):

| Env var | Config key |
|---|---|
| `UAII_LOG_LEVEL` | `log.level` |
| `UAII_LOG_COLOR` | `log.color` |
| `UAII_PLUGIN__DIRS` | `plugin.dirs` |

## Plugin ABI (Phase 1)

Every plugin dynamic library must export:

- `uaii_plugin_get_info`
- `uaii_plugin_init`
- `uaii_plugin_shutdown`

See [`include/uaii/c_api/plugin_abi.h`](include/uaii/c_api/plugin_abi.h) and
[`plugins/example_probe`](plugins/example_probe).

Host rejects plugins whose `abi_version != UAII_PLUGIN_ABI_VERSION`.

## Phase 1 exit criteria

- [x] CMake workspace with modular `uaii-*` libraries
- [x] Plugin discovery + load with ABI gate
- [x] Logging, config, structured errors
- [x] Core interfaces declared (`IBackend`, loaders, operators, storage, scheduler, tokenizer)
- [x] `uaii doctor` reports environment
- [x] Open-source project metadata (license, contributing, security, CI)

## Contributing & community

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- License: [MIT](LICENSE)

## License

MIT © 2026 arjun shukla and contributors
