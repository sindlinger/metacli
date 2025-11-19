#!/usr/bin/env bash
set -euo pipefail
# recipes/indicator_full.sh — ciclo completo: install+compile → attach → validate
tools/agentctl listener start EURUSD H1
tools/agentctl indicator full "templates/Indicator_Sample.mq5" --symbol EURUSD --tf H1 --subwin 1
