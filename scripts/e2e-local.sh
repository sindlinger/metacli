#!/usr/bin/env bash
# Teste E2E local via arquivos cmd/resp (não abre UI automaticamente).
# Pré-requisitos: MT5/CommandListener rodando com data_dir configurado e g_files_dir acessível.

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

DATA_DIR=${DATA_DIR:-"$ROOT/mql5/Files"}
FILES_DIR="$DATA_DIR"

if [ ! -d "$FILES_DIR" ]; then
  echo "[e2e] Files dir não encontrado: $FILES_DIR" >&2
  exit 1
fi

send_cmd() {
  local type="$1"; shift
  local params=()
  while (("$#")); do params+=("$1"); shift; done
  local id="test$(date +%s%3N)"
  local cmd="$id|$type"
  for p in "${params[@]}"; do cmd+="|$p"; done
  local cmd_file="$FILES_DIR/cmd_${id}.txt"
  local resp_file="$FILES_DIR/resp_${id}.txt"
  echo -e "$cmd" > "$cmd_file"
  echo "[e2e] >> $cmd"
  for _ in {1..50}; do
    if [ -f "$resp_file" ]; then
      echo "[e2e] << $(cat "$resp_file")"
      rm -f "$resp_file"
      return 0
    fi
    sleep 0.2
  done
  echo "[e2e] timeout aguardando $resp_file" >&2
  return 1
}

send_cmd PING

