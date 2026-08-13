#!/usr/bin/env bash
# Build a UAII Python wheel with bundled native uaii_capi (Linux/macOS).
# Run from repo root.
#
# Prerequisites: CMake, a C++ toolchain, Python 3.10+ with pip.
#
# Usage:
#   bash bindings/python/scripts/build_wheel.sh
#   BUILD_DIR=build CONFIG=Release bash bindings/python/scripts/build_wheel.sh
#
# Tip: pass -DUAII_BUILD_PYTHON=ON to cmake if you also want the pybind extension.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PY_ROOT/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
CONFIG="${CONFIG:-Release}"

cd "$REPO_ROOT"

echo "==> cmake configure (tip: -DUAII_BUILD_PYTHON=ON for pybind)"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG" -DUAII_BUILD_PLUGINS=ON

echo "==> cmake --build $BUILD_DIR --config $CONFIG"
cmake --build "$BUILD_DIR" --config "$CONFIG" --parallel

echo "==> bundle_native.py"
python3 "$PY_ROOT/scripts/bundle_native.py" --build-dir "$BUILD_DIR"

echo "==> python -m build"
python3 -m pip install -q build
python3 -m build "$PY_ROOT"

echo "Wheel(s) under: $PY_ROOT/dist"
