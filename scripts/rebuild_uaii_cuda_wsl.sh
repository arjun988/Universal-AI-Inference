#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:/usr/bin:/bin:${PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Prefer an explicit override, then a linux-native clone, then this script's repo.
if [[ -n "${UAII_SRC:-}" ]]; then
  SRC="$UAII_SRC"
elif [[ -f "$HOME/Universal-AI-Inference/CMakeLists.txt" ]]; then
  SRC="$HOME/Universal-AI-Inference"
elif [[ -f "$ROOT/CMakeLists.txt" ]]; then
  SRC="$ROOT"
elif [[ -f /mnt/c/Users/Arjun/Desktop/Universal-AI-Inference/CMakeLists.txt ]]; then
  SRC="/mnt/c/Users/Arjun/Desktop/Universal-AI-Inference"
else
  echo "error: cannot find UAII source (set UAII_SRC)" >&2
  exit 1
fi
BUILD="${UAII_WSL_BUILD:-$HOME/uaii-wsl-build-dash}"
CMAKE_BIN="${CMAKE_BIN:-$HOME/cmake-3.30.5-linux-x86_64/bin/cmake}"
if [[ ! -x "$CMAKE_BIN" ]]; then CMAKE_BIN="$(command -v cmake)"; fi

echo "SRC=$SRC"
echo "BUILD=$BUILD"
echo "CMAKE=$CMAKE_BIN"
nvcc --version
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true

# Detect whether nvcc understands Blackwell (sm_120). CUDA 12.6 does not; 12.8+ does.
ARCHS="${CMAKE_CUDA_ARCHITECTURES:-}"
if [[ -z "$ARCHS" ]]; then
  if echo 'int main(){return 0;}' | nvcc -x cu - -o /tmp/uaii_nvcc_sm120_probe --generate-code=arch=compute_120,code=sm_120 >/dev/null 2>&1; then
    ARCHS="75;80;86;89;120"
  else
    echo "note: nvcc cannot target sm_120 (need CUDA toolkit >= 12.8 for RTX 50-series)."
    echo "      Building native CUDA for sm_75–89; upgrade toolkit for full Blackwell support."
    ARCHS="75;80;86;89"
  fi
fi
rm -f /tmp/uaii_nvcc_sm120_probe

mkdir -p "$BUILD"
# Clear stale CUDA arch / separable-comp flags from prior configures.
rm -f "$BUILD/CMakeCache.txt"
"$CMAKE_BIN" -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUAII_WITH_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS"

"$CMAKE_BIN" --build "$BUILD" --target uaii --parallel "$(nproc 2>/dev/null || echo 8)"

BIN="$BUILD/libs/uaii-cli/uaii"
echo "Built: $BIN"
"$BIN" version
"$BIN" doctor 2>&1 | sed -n '/Backends/,/Interfaces/p'
