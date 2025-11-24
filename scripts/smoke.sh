#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "[smoke] build"
npm run --silent build

echo "[smoke] cli help"
node dist/cli.js --help >/dev/null

DATA_DIR=${DATA_DIR:-${MTCLI_DATA_DIR:-}}
if [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
  echo "[smoke] e2e PING via cmd/resp"
  DATA_DIR="$DATA_DIR" ./scripts/e2e-local.sh || { echo "[smoke] e2e failed"; exit 1; }
else
  echo "[smoke] e2e skipped (DATA_DIR not set or missing)"
fi

echo "[smoke] done"
