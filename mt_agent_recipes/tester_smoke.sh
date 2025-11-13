#!/usr/bin/env bash
set -euo pipefail
tools/agentctl tester smoke "Examples\\MACD\\MACD Sample" EURUSD M1 --visual --report \reports\smoke_{ts}.htm --replace-report --shutdown
