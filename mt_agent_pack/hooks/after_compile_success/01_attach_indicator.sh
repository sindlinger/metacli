#!/usr/bin/env bash
set -euo pipefail
# Hook: after_compile_success -> anexa indicador recém-compilado no EURUSD H1 subwin 1
if [[ "${MTCLI_TYPE:-}" == "indicator" ]]; then
  name="${MTCLI_NAME%.mq5}"
  echo "[hook] Anexando indicador compilado: ${name} em EURUSD H1 (subwin 1)"
  mtcli chart indicator attach --symbol EURUSD --period H1 --indicator "$name" --subwindow 1 || true
fi
