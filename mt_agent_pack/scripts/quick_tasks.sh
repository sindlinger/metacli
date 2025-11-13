#!/usr/bin/env bash
set -euo pipefail
# scripts/quick_tasks.sh — utilidades rápidas para o agente

cmd="${1:-}"; shift || true

case "$cmd" in
  install_indicator)
    file="${1:?Uso: $0 install_indicator <arquivo.mq5|.ex5>}"
    mtcli install indicator --file "$file"
    ;;
  install_expert)
    file="${1:?Uso: $0 install_expert <arquivo.mq5|.ex5>}"
    mtcli install expert --file "$file"
    ;;
  attach_indicator)
    name="${1:?nome do indicador (sem .mq5)}"; sym="${2:-EURUSD}"; tf="${3:-H1}"; sub="${4:-1}"
    mtcli chart indicator attach --symbol "$sym" --period "$tf" --indicator "$name" --subwindow "$sub"
    ;;
  detach_indicator)
    name="${1:?nome do indicador}"; sym="${2:-EURUSD}"; tf="${3:-H1}"; sub="${4:-1}"
    mtcli chart indicator detach --symbol "$sym" --period "$tf" --indicator "$name" --subwindow "$sub"
    ;;
  attach_expert)
    rel="${1:?ex.: Pasta\\MeuEA}"; sym="${2:-EURUSD}"; tf="${3:-H1}"; tpl="${4:-}"
    if [[ -n "$tpl" ]]; then
      mtcli chart expert attach --symbol "$sym" --period "$tf" --expert "$rel" --template "$tpl"
    else
      echo "[i] Sem --template informado; se tiver um .tpl local, copie com --template-src via mtcli chart expert attach"
    fi
    ;;
  detach_expert)
    sym="${1:-EURUSD}"; tf="${2:-H1}"
    mtcli chart expert detach --symbol "$sym" --period "$tf"
    ;;
  tester_run)
    ea="${1:?RelPath do EA}"; sym="${2:-EURUSD}"; tf="${3:-M1}"
    shift 3 || true
    mtcli tester run --ea "$ea" --symbol "$sym" --period "$tf" "$@"
    ;;
  *)
    echo "Comandos: install_indicator | install_expert | attach_indicator | detach_indicator | attach_expert | detach_expert | tester_run"
    exit 1
    ;;
esac
