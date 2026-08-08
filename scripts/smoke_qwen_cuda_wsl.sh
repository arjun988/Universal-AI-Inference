#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-$HOME/uaii-wsl-build-dash/libs/uaii-cli/uaii}"
MODEL="${2:-$HOME/uaii-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"

echo "== tokenize =="
"$BIN" tokenize encode "Hi" --gguf "$MODEL"
"$BIN" tokenize decode "$("$BIN" tokenize encode "Hi" --gguf "$MODEL" | tr -d '\r')" --gguf "$MODEL"
echo
"$BIN" tokenize encode "<|im_start|>" --gguf "$MODEL"

CMDS="$(mktemp)"
cat >"$CMDS" <<'EOF'
{"cmd":"generate","id":"1","prompt":"Say hello in one short sentence.","max_new_tokens":32,"stream":false,"temperature":0}
{"cmd":"quit"}
EOF
echo "== generate =="
"$BIN" chat --jsonl --backend cuda --model "$MODEL" --log-level error --max-context 256 <"$CMDS" \
  | awk '/"event":"(ready|done|error|bye)"/'
rm -f "$CMDS"
