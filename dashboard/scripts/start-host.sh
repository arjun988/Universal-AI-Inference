#!/usr/bin/env bash
# Self-host dashboard (LAN). Requires UAII_DASH_TOKEN.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${UAII_DASH_TOKEN:-}" ]]; then
  echo "Set UAII_DASH_TOKEN before self-hosting, e.g.:"
  echo "  export UAII_DASH_TOKEN=\$(openssl rand -hex 16)"
  exit 1
fi
if [[ ! -d node_modules ]]; then npm run install:all; fi
npm run build
export NODE_ENV=production
export UAII_DASH_BIND="${UAII_DASH_BIND:-0.0.0.0}"
export UAII_DASH_PORT="${UAII_DASH_PORT:-8787}"
echo "Self-host listening on http://${UAII_DASH_BIND}:${UAII_DASH_PORT}"
echo "Auth: Bearer \$UAII_DASH_TOKEN"
exec npm start
