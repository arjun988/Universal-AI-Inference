#!/usr/bin/env bash
# Build+run uaii_bench inside WSL (avoids Windows Application Control on unsigned .exe).
#
# Env:
#   TRIALS=21|100     (default 21)
#   SUITE=all|gemm,...
#   PROVIDERS=all|ref,onednn,openblas
#   UAII_BENCH_CPU=...
#   UAII_WSL_BUILD=...  override build dir
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$ROOT" == /mnt/c/* ]]; then
  BUILD="${UAII_WSL_BUILD:-$HOME/uaii-wsl-build-vendor}"
  CMAKE_BIN="${CMAKE_BIN:-$HOME/cmake-3.30.5-linux-x86_64/bin/cmake}"
else
  BUILD="${UAII_WSL_BUILD:-$ROOT/build-wsl-vendor}"
  CMAKE_BIN="${CMAKE_BIN:-cmake}"
fi

export PATH="$(dirname "$CMAKE_BIN"):${PATH:-}"

if [[ ! -x "$CMAKE_BIN" ]] && ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Install cmake or set CMAKE_BIN." >&2
  exit 1
fi
CMAKE_BIN="$(command -v cmake || true)"
if [[ -x "${HOME}/cmake-3.30.5-linux-x86_64/bin/cmake" ]]; then
  CMAKE_BIN="${HOME}/cmake-3.30.5-linux-x86_64/bin/cmake"
fi

TRIALS="${TRIALS:-21}"
SUITE="${SUITE:-all}"
PROVIDERS="${PROVIDERS:-all}"
WARMUP="${WARMUP:-5}"

# Soft-install vendor deps when sudo works without a password.
if command -v apt-get >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    sudo -n apt-get update -y >/dev/null 2>&1 || true
    sudo -n apt-get install -y libdnnl-dev libopenblas-dev >/dev/null 2>&1 || true
  fi
fi

ONEDNN=OFF
OPENBLAS=OFF
if [[ -f /usr/include/dnnl/dnnl.hpp ]] || pkg-config --exists dnnl 2>/dev/null; then
  ONEDNN=ON
fi
if ldconfig -p 2>/dev/null | grep -q libopenblas || [[ -f /usr/include/openblas/cblas.h ]] || \
   [[ -f /usr/include/x86_64-linux-gnu/cblas.h ]]; then
  OPENBLAS=ON
fi
# Also try common library names
if [[ -e /usr/lib/x86_64-linux-gnu/libopenblas.so ]] || [[ -e /usr/lib/libopenblas.so ]]; then
  OPENBLAS=ON
fi

echo "Configure: UAII_WITH_ONEDNN=${ONEDNN} UAII_WITH_OPENBLAS=${OPENBLAS}"
"$CMAKE_BIN" -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUAII_BUILD_TESTS=OFF \
  -DUAII_BUILD_PLUGINS=OFF \
  -DUAII_BUILD_BENCHMARKS=ON \
  -DUAII_WITH_ONEDNN="${ONEDNN}" \
  -DUAII_WITH_OPENBLAS="${OPENBLAS}"
"$CMAKE_BIN" --build "$BUILD" --target uaii_bench --parallel "$(nproc)"

if [[ -z "${UAII_BENCH_CPU:-}" ]]; then
  if [[ -f /proc/cpuinfo ]]; then
    UAII_BENCH_CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^[[:space:]]*//')"
    export UAII_BENCH_CPU
  fi
fi

mkdir -p "$ROOT/benchmarks/results"
OUT="$ROOT/benchmarks/results/local_wsl.json"
echo "CPU=${UAII_BENCH_CPU:-unknown}"
echo "Running → $OUT (suite=${SUITE} providers=${PROVIDERS} trials=${TRIALS})"
"$BUILD/benchmarks/uaii_bench" \
  --suite "$SUITE" \
  --providers "$PROVIDERS" \
  --trials "$TRIALS" \
  --warmup "$WARMUP" \
  --json | tee "$OUT"
echo "Wrote $OUT"
