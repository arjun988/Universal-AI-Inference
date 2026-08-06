#!/usr/bin/env bash
# Build+run uaii_bench inside WSL (avoids Windows Application Control on unsigned .exe).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer Linux home build dir when invoked from /mnt/c (faster + no WDAC).
if [[ "$ROOT" == /mnt/c/* ]]; then
  BUILD="${UAII_WSL_BUILD:-$HOME/uaii-wsl-build}"
  CMAKE_BIN="${CMAKE_BIN:-$HOME/cmake-3.30.5-linux-x86_64/bin/cmake}"
else
  BUILD="${UAII_WSL_BUILD:-$ROOT/build-wsl}"
  CMAKE_BIN="${CMAKE_BIN:-cmake}"
fi

export PATH="$(dirname "$CMAKE_BIN"):$PATH"

if [[ ! -x "$CMAKE_BIN" ]] && ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Install cmake or set CMAKE_BIN." >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUAII_BUILD_TESTS=OFF \
  -DUAII_BUILD_PLUGINS=OFF \
  -DUAII_BUILD_BENCHMARKS=ON
cmake --build "$BUILD" --target uaii_bench --parallel "$(nproc)"

if [[ -z "${UAII_BENCH_CPU:-}" ]]; then
  if [[ -f /proc/cpuinfo ]]; then
    UAII_BENCH_CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^[[:space:]]*//')"
    export UAII_BENCH_CPU
  fi
fi

mkdir -p "$ROOT/benchmarks/results"
OUT="$ROOT/benchmarks/results/local_wsl.json"
echo "CPU=${UAII_BENCH_CPU:-unknown}"
echo "Running → $OUT"
"$BUILD/benchmarks/uaii_bench" --trials 21 --warmup 5 --json | tee "$OUT"
echo "Wrote $OUT"
