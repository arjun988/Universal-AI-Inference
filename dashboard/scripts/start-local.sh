#!/usr/bin/env bash
# Local dashboard (loopback). Builds UI if needed, starts API+static on :8787
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d node_modules ]]; then npm run install:all; fi
npm run build
export NODE_ENV=production
export UAII_DASH_BIND="${UAII_DASH_BIND:-127.0.0.1}"
export UAII_DASH_PORT="${UAII_DASH_PORT:-8787}"
echo "Opening local dashboard at http://${UAII_DASH_BIND}:${UAII_DASH_PORT}"
exec npm start
