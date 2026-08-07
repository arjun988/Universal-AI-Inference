#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${UAII_WSL_BUILD:-$HOME/uaii-wsl-build-dash}"
CMAKE_BIN="${CMAKE_BIN:-$HOME/cmake-3.30.5-linux-x86_64/bin/cmake}"
if [[ ! -x "$CMAKE_BIN" ]]; then CMAKE_BIN="$(command -v cmake)"; fi
mkdir -p "$BUILD"
cd "$ROOT"
"$CMAKE_BIN" -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DUAII_BUILD_BENCHMARKS=OFF -DUAII_BUILD_TESTS=OFF
"$CMAKE_BIN" --build "$BUILD" --target uaii --parallel
echo "OK: $BUILD/libs/uaii-cli/uaii"
