#!/usr/bin/env bash
set -euo pipefail
# bootstrap_mt_agent.sh — prepara o ambiente do MT5 para automação
# Uso: ./bootstrap_mt_agent.sh /caminho/para/pasta_mtcli

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 /caminho/para/pasta_mtcli"
  exit 1
fi

MTCLI_PATH="$1"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[*] MTCLI em: $MTCLI_PATH"

# 1) Patch no mtcli (se tiver patch kit ao lado)
if [[ -f "./mtcli_patchkit.zip" ]]; then
  echo "[*] Aplicando mtcli_patchkit.zip"
  unzip -o "./mtcli_patchkit.zip" >/dev/null
  chmod +x apply_mtcli_patch.sh
  ./apply_mtcli_patch.sh "$MTCLI_PATH"
else
  echo "[i] mtcli_patchkit.zip não encontrado aqui. Pulando patch."
fi

# 2) Instalar hooks locais (além dos do patch kit)
HOOK_DIR="${HOME}/.mtcli/hooks.d/after_compile_success"
mkdir -p "$HOOK_DIR"
cp -f "${BASE_DIR}/hooks/after_compile_success/01_attach_indicator.sh" "$HOOK_DIR/01_attach_indicator_local.sh"
chmod +x "$HOOK_DIR/01_attach_indicator_local.sh"
echo "[*] Hook local instalado em $HOOK_DIR/01_attach_indicator_local.sh"

# 3) Instalar CLI slash-mql5 (se zip presente)
if [[ -f "./slash-mql5-cli_with_docs.zip" ]]; then
  echo "[*] Instalando slash-mql5-cli"
  rm -rf ./slash-mql5-cli
  unzip -o "./slash-mql5-cli_with_docs.zip" >/dev/null
  (cd slash-mql5-cli && python3 -m pip install --user pipx && pipx install . || python3 -m pip install .)
  # opcional: baixar docs
  echo "/doc pull" | slash-mql5 || true
else
  echo "[i] slash-mql5-cli_with_docs.zip não encontrado aqui. Pulando instalação do CLI."
fi

# 4) Criar config de exemplo do mtcli
mkdir -p "${HOME}/.mtcli"
if [[ ! -f "${HOME}/.mtcli/config.json" ]]; then
  cp -f "${BASE_DIR}/config/example_config.json" "${HOME}/.mtcli/config.json"
  echo "[*] Exemplo de config copiado para ~/.mtcli/config.json (ajuste os caminhos)."
else
  echo "[i] ~/.mtcli/config.json já existe; revise manualmente se precisar."
fi

echo "[ok] Bootstrap finalizado."
