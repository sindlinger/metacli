# MT Agent Pack — Ambiente de automação para MetaTrader 5

Este pacote cria uma **pasta de trabalho** para um agente (Codex ou equivalente) operar no **MetaTrader 5** com passos determinísticos:
- instalar/compilar **Indicators/Experts**,
- anexar/remover no gráfico via **CommandListenerEA**,
- ler **logs**,
- rodar **Strategy Tester**,
- consultar **documentação** (via `slash-mql5` com `/doc pull|search|open`),
- e usar **hooks** para ações pós-compilação (ex.: anexar em EURUSD/H1 automaticamente).

> Pré-requisitos (no Windows/WSL):
> - Python 3.9+
> - `mtcli.py` (seu CLI MT5) acessível
> - MetaTrader 5 instalado (terminal64.exe/metaeditor64.exe)
> - A Data Folder configurada
> - (Opcional) Pipx para instalar `slash-mql5-cli`

## Estrutura
```
mt-agent/
  README.md
  bootstrap_mt_agent.sh     # prepara tudo (patch mtcli, hooks, slash-mql5, docs)
  agent/mt_agent_e2e.txt    # prompt de agente "faça já", sabendo o que rodar
  hooks/after_compile_success/01_attach_indicator.sh  # exemplo: EURUSD/H1
  scripts/quick_tasks.sh    # tarefas rápidas (install/attach/tester/logs)
  templates/Indicator_Sample.mq5
  templates/EA_Sample.mq5
  config/example_config.json
```

## Uso rápido
1) **Crie a pasta de trabalho** e extraia este ZIP dentro dela.
2) Rode o bootstrap (ajuste caminhos quando solicitado):
```bash
chmod +x bootstrap_mt_agent.sh
./bootstrap_mt_agent.sh /caminho/para/mtcli.py
```
3) Diga ao agente para usar `agent/mt_agent_e2e.txt` como prompt principal.

## Tarefas comuns (scripts/quick_tasks.sh)
```bash
# instalar e compilar um indicador
./scripts/quick_tasks.sh install_indicator templates/Indicator_Sample.mq5

# anexar indicador no EURUSD/H1 (subwin 1)
./scripts/quick_tasks.sh attach_indicator "Indicator_Sample" EURUSD H1 1

# instalar e compilar um expert
./scripts/quick_tasks.sh install_expert templates/EA_Sample.mq5

# anexar expert com template existente
./scripts/quick_tasks.sh attach_expert "MinhaPasta\MeuEA" EURUSD H1 MeuEA.tpl

# rodar tester visual simples
./scripts/quick_tasks.sh tester_run "Examples\MACD\MACD Sample" EURUSD M1 --visual
```

## Observações
- **Sem criar .tpl do zero**: use `--template` (já existente) ou `--template-src` (ele copia).
- **Compilação**: falha cedo com log `.log` ao lado do fonte.
- **Hooks**: edite `hooks/after_compile_success/01_attach_indicator.sh` para trocar símbolo/timeframe/subjanela.
