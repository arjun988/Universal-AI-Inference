#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${UAII_WSL_BUILD:-$HOME/uaii-wsl-build-dash}"
CMAKE_BIN="${CMAKE_BIN:-$HOME/cmake-3.30.5-linux-x86_64/bin/cmake}"
if [[ ! -x "$CMAKE_BIN" ]]; then CMAKE_BIN="$(command -v cmake)"; fi
MODEL="${1:-$ROOT/dashboard/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"
"$CMAKE_BIN" --build "$BUILD" --target uaii --parallel 8
BIN="$BUILD/libs/uaii-cli/uaii"
"$BIN" version
printf '%s\n' \
  '{"cmd":"generate","id":"1","prompt":"Say hello in one short sentence.","max_new_tokens":16,"stream":false}' \
  '{"cmd":"quit"}' \
  | "$BIN" chat --jsonl --model "$MODEL" --log-level error --max-context 256
