# Changelog — mtcli

# Changelog — mtcli

## 2025-11-20

### Diagnósticos de chart/indicator (Chart* APIs)
- `chart close`, `chart window-find` e `chart drop-info` expõem direto `ChartClose`, `ChartWindowFind` e `ChartWindowOnDropped`, respeitando os defaults do projeto.
- `chart template save` envia `ChartSaveTemplate` (nome automático `<symbol>-<period>-YYYYMMDD-HHMMSS.tpl` se não informado).
- `chart screenshot sweep` dispara séries automáticas pelo `SCREENSHOT_SWEEP` (ChartNavigate + ChartScreenShot), com controle de passos/alinhamento/formato e cópia opcional para um diretório local.
- Novos utilitários `indicator total`, `indicator name` e `indicator handle` fazem ponte com `ChartIndicatorsTotal`, `ChartIndicatorName` e `ChartIndicatorGet` para depurar janelas específicas.
- Listener (`CommandListenerEA`) ganhou os comandos `CLOSE_CHART`, `WINDOW_FIND`, `DROP_INFO`, `IND_TOTAL`, `IND_NAME`, `IND_HANDLE` e `SAVE_TPL`, mantendo os logs centralizados no `Experts` log.

## 2025-11-13

### Interface reorganizada e execução não bloqueante
- `terminal`: status/verify-datapath/screenshot/chart (indicator|expert|template)/create/tester/kill.
- `dll`: build/build-all/link/ping/fft/heartbeat/package/fetch-cuda/list/gen-mql.
- `metaeditor`: compile.
- `tester`: alias de `terminal tester run`.
- `project`: init|save|use|show|list|defaults set|show + detect/env (movidos do topo).
- Comandos legados continuam disponíveis, mas foram suprimidos do help principal.

### Portability/datapath automáticos
- Todos os comandos MT aceitam `--portable 0|1` e `--profile` (também via `project defaults set`).
- Quando o usuário não especifica nada, o CLI compara o `data_dir` salvo com o diretório do `terminal64.exe` e decide automaticamente:
  - se `data_dir == <terminal_dir>\MQL5` → usa `/portable`.
  - caso contrário → usa `/datapath:<raiz>` (pai do MQL5), garantindo que o Terminal abra exatamente na pasta escolhida (inclusive AppData ou unidades diferentes).
- Se um default antigo marcar `portable=1` mas o projeto apontar para outra pasta, o CLI ignora o `/portable` e força `/datapath`, evitando quedas involuntárias no AppData.
- `listener start --trace` agora exibe claramente a linha com `/portable` ou `/datapath`, facilitando auditoria.

### Não bloqueante + logs
- Todos os comandos que tocam o MT retornam imediatamente com um tail curto e um `[done]` (`listener start/status`, `terminal chart attach|detach`, `screenshot`, `install indicator|expert|template|script`, `tester run`, `verify-datapath`, `logs tail`).

### Outras notas
- `config lang set|show` substitui o antigo `mtcli lang ...`.
- `project detect` lista Terminals/MQL5; `project env` mostra diagnóstico (WSL/Interop/Python).
- `collect_log_targets` bugfix (parêntese) para evitar falhas de leitura.

Use estes passos para validar rapidamente:
1. `mtcli project init --yes`
2. `mtcli terminal status --repair`
3. `mtcli terminal chart indicator attach --indicator <Nome> --symbol EURUSD --period H1 --subwindow 1`
4. `mtcli terminal screenshot --symbol EURUSD --period H1`
5. `mtcli terminal chart indicator detach --indicator <Nome>`
6. `mtcli terminal tester run --expert <EA> --symbol EURUSD --period H1 --visual off`
