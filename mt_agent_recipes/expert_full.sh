#!/usr/bin/env bash
set -euo pipefail
# recipes/expert_full.sh — ciclo completo para EA (se tiver tpl)
tools/agentctl listener start EURUSD H1
tools/agentctl expert full "templates/EA_Sample.mq5" --symbol EURUSD --tf H1 --tpl MeuEA.tpl || echo "[i] Sem tpl: attach pulado"
