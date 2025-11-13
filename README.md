# Gen2 MTCLI (Standalone)

CLI agnóstico para automação do MetaTrader 5 dentro de projetos. Pode ser copiado para qualquer repositório.

## Estrutura recomendada

```text
<repo>/
  mtcli/            # este diretório (mtcli.py, agentctl, etc.)
  bin/mtcli         # wrapper Bash (Linux/WSL)
  bin/mtcli.cmd     # wrapper Windows (CMD)
```

- Os wrappers chamam `python3 mtcli/mtcli.py` (ou `venv/bin/python` se existir um venv).
- Você pode apontar para outra localização usando a variável de ambiente `MTCLI_ROOT`.

## Uso rápido

```bash
# via wrapper
./bin/mtcli
./bin/mtcli project init --yes
./bin/mtcli mt attach --indicator MeuIndicador

# direto
python3 mtcli/mtcli.py mt status --repair
```

## Estado por projeto

- Cada repositório mantém seu `mtcli_projects.json` (por padrão em `mtcli/mtcli_projects.json`).
- O `last_project` selecionado vale apenas para o repo onde você está executando.

## Integrações úteis

- Terminal: `/config:<ini>`, `/profile:<nome>`, `/portable`. O mtcli gera INI (StartUp) e inicia destacado.
- MetaEditor: `/compile`, `/log`, `/s` (syntax-check), `/include:<pasta>`.

## Dual-mode (MT aberto ou fechado)

- Comandos `mt attach/detach/apply` operam via listener quando ativo; senão, geram INI + `cmd.txt` e iniciam o Terminal.
- Sempre retornam um tail de logs padronizado.
# MTCLI
## I18N (Language)

- Current language is stored in `tools/mtcli_config.json` under key `lang`.
- Switch language at runtime:
  - Show: `mtcli lang show`
  - Set English: `mtcli lang set --to en`
  - Set Portuguese: `mtcli lang set --to pt`

### Developer guidelines

- When adding messages, never print raw strings directly.
- Use the translation helper `tr(key, **kwargs)` and define the key in both dictionaries:
  - `I18N['en'][key] = '…'`
  - `I18N['pt'][key] = '…'`
- Prefer concise, action‑oriented messages. Keep placeholders named and explicit (e.g., `{path}`, `{limit}`, `{rc}`).
- If a key is missing, the CLI falls back to English.
