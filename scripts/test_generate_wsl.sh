#!/usr/bin/env bash
# Build uaii in WSL and smoke-test `uaii generate` (avoids Windows App Control).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${UAII_WSL_BUILD:-$HOME/uaii-wsl-build-dash}"
CMAKE_BIN="${CMAKE_BIN:-$HOME/cmake-3.30.5-linux-x86_64/bin/cmake}"
if [[ ! -x "$CMAKE_BIN" ]]; then
  CMAKE_BIN="$(command -v cmake)"
fi
mkdir -p "$BUILD"
cd "$ROOT"
"$CMAKE_BIN" -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DUAII_BUILD_BENCHMARKS=OFF -DUAII_BUILD_TESTS=OFF
"$CMAKE_BIN" --build "$BUILD" --target uaii --parallel
echo "=== generate --demo ==="
"$BUILD/libs/uaii-cli/uaii" generate --demo --prompt "hello" --max-new-tokens 4 --json --log-level error
echo "=== chat jsonl one-shot ==="
printf '%s\n' '{"cmd":"generate","id":"1","prompt":"hi","max_new_tokens":3,"stream":false}' '{"cmd":"quit"}' \
  | "$BUILD/libs/uaii-cli/uaii" chat --demo --jsonl --log-level error
