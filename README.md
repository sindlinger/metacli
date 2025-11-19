# Gen2 MTCLI (Standalone)

CLI agnóstico para automação do MetaTrader 5 dentro de projetos. Pode ser copiado para qualquer repositório.

## CLI em TypeScript

- Implementação em TypeScript (`src/*.ts`) com build via `npm run build`.
- Executável único `./bin/mtcli.js` (funciona em WSL e Windows). Exemplos:

  ```bash
  npm install
  npm run build
  ./bin/mtcli.js project init            # cria projeto padrão (Dukascopy)
  ./bin/mtcli.js listener run            # abre o terminal com listener.ini
  ./bin/mtcli.js chart indicator attach  # usa defaults do projeto
  ```

- `project init` já configura terminal/metaeditor/data_dir e reinicia o listener automaticamente (use `--no-start-listener` para pular).
- Comandos `chart indicator/template` reiniciam o listener antes de escrever no `cmd.txt`, garantindo que o MT5 esteja pronto.

- Grupos de comandos disponíveis:
  - `project`: init/save/show/defaults…
  - `listener`: run/status/ensure…
  - `chart`: indicator attach/detach, template apply (com --file ou --name)…
  - `tester`, `editor`, `dll`, `utils`, `config`: stubs prontos para expansão.
- Todos os comandos leem/escrevem o mesmo `mtcli_projects.json` (compatível com o script Python legado).

## Estrutura recomendada

```text
<repo>/
  mtcli/            # este diretório (código TS + scripts legados)
  bin/mtcli         # wrapper Bash (Linux/WSL)
  bin/mtcli.cmd     # wrapper Windows (CMD)
```

- Os wrappers podem chamar `./bin/mtcli.js` diretamente ou `python3 mtcli.py` (legado).
- Você pode apontar para outra localização usando a variável de ambiente `MTCLI_ROOT`.

## Uso rápido

```bash
# inicializa projeto usando os caminhos padrão
./bin/mtcli.js project init

# anexa um indicador usando defaults (EURUSD/H1/subwindow=1)
./bin/mtcli.js chart indicator attach --indicator WaveSpecZZ_Project/WaveSpecZZ_1.1.0-gpuopt

# aplica template diretamente (aceita --file ou --name já existente)
./bin/mtcli.js chart template apply --file templates/Wave.tpl
```

## Estado por projeto

- Cada repositório mantém seu `mtcli_projects.json` (por padrão em `mtcli/mtcli_projects.json`).
- O `last_project` selecionado vale apenas para o repo atual.

## Integrações úteis

- Chart/listener: `/config:<ini>`, `/profile:<nome>`, `/datapath:<pasta>` — o CLI gera o INI e inicia destacado.
- MetaEditor (placeholder): `/compile`, `/log`, `/include:<pasta>`.

## Dual-mode (MT aberto ou fechado)

- Comandos `chart indicator/template` operam via listener quando ativo; senão, basta iniciar o terminal (`listener run`).
- Sempre retorne logs/screenshots para validar antes e depois.

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
