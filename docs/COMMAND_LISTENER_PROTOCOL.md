# CommandListener Protocol (listener v1.0.6)

Arquivos: o CLI cria `cmd_<id>.txt` em `MQL5/Files` e espera `resp_<id>.txt`.

Formato de cmd: `id|TYPE|param1|param2|...` (linhas LF)
Formato de resp: 1ª linha `OK` ou `ERROR`; 2ª mensagem; demais linhas `data`.

Tipos suportados (EA `CommandListener.mq5`):

- **PING** → resp: `pong <version>`
- **DEBUG_MSG text**
- **GLOBAL_SET name value** / **GLOBAL_GET name** / **GLOBAL_DEL name** /
  **GLOBAL_DEL_PREFIX prefix** / **GLOBAL_LIST [prefix] [limit]**
- **OPEN_CHART symbol period**
- **CLOSE_CHART symbol period** / **CLOSE_ALL**
- **APPLY_TPL symbol period tpl** / **SAVE_TPL symbol period tpl**
- **REDRAW_CHART [chart_id]**
- **WINDOW_FIND symbol period indicator_name**
- **DROP_INFO** (info do chart atual)
- **SCREENSHOT symbol period file [width] [height]**
- **SCREENSHOT_SWEEP symbol period folder base steps shift align width height fmt delay**
- **ATTACH_IND_FULL symbol period name subwindow params**
- **DETACH_IND_FULL symbol period name subwindow**
- **IND_TOTAL symbol period subwindow** / **IND_NAME symbol period subwindow index** /
  **IND_HANDLE symbol period subwindow name**
- **ATTACH_EA_FULL symbol period expert template params**
- **DETACH_EA_FULL**
- **LIST_CHARTS**
- **LIST_INPUTS** (retorna os pares do último attach) / **SET_INPUT name value** (reattacha)
- **SNAPSHOT_SAVE name** / **SNAPSHOT_APPLY name** / **SNAPSHOT_LIST**
- **OBJ_LIST** / **OBJ_DELETE name** / **OBJ_DELETE_PREFIX prefix** /
  **OBJ_MOVE name time price** /
  **OBJ_CREATE type name payload** (payload `key=value;...`, suporta time/price/time2/price2/text/color/style/width/anchor/x/y/fontsize/font/back/selectable/hidden)
- **TRADE_BUY symbol volume [sl] [tp] [comment]**
- **TRADE_SELL symbol volume [sl] [tp] [comment]**
- **TRADE_CLOSE_ALL** / **TRADE_LIST**

Notas de params:
- `period`: M1/M5/M15/M30/H1/H4/D1/W1/MN1
- `params` (ind/ea): `k=v;k2=v2` (num vira double; senão string)
- `align` em SCREENSHOT_SWEEP: `left` ou outro (direita)
- cores em OBJ_CREATE: inteiro ARGB (StringToInteger)

Status da versão 1.0.6:
- Inputs reattach via `SET_INPUT` e persistem em memória do listener.
- EA detach remove indicadores `Experts\*` ou aplica Default.tpl se existir.
- Trade usa CTrade, valida símbolo e volume > 0.

