#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-/mnt/c/Users/Arjun/Desktop/Universal-AI-Inference/dashboard/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf}"
URL="https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
  sz=$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST")
  if (( sz > 600000000 )); then
    ls -lh "$DEST"
    echo "Already downloaded."
    exit 0
  fi
fi

echo "Downloading TinyLlama-1.1B-Chat Q4_K_M (~670 MB)..."
echo "  -> $DEST"
curl -L --fail --retry 3 --retry-delay 5 -o "${DEST}.part" "$URL"
mv "${DEST}.part" "$DEST"
ls -lh "$DEST"
echo "Done."
