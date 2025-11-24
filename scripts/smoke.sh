#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "[smoke] build"
npm run --silent build

echo "[smoke] cli help"
node dist/cli.js --help >/dev/null

echo "[smoke] done"
