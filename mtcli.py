#!/usr/bin/env python3
r"""
mtcli.py — MetaTrader helper CLI for ALGLIB bridge

Commands:
  detect                 Detect Terminal folders (MQL5\Libraries) for current user
  list                   List dist-wave artifacts with times and hashes
  build                  Configure+build via CMake; optional preset regen
  build-all              Generate (glue+exports) then build Core, Service and Bridge (Release)
  link                   Link the project’s MQL5\\Libraries (or explicit --libs/--all) to build-win\\Release (no file copies)
  kill                   Kill running processes (terminal/metaeditor/alglib_worker)

Examples (PowerShell, one per line):
  python tools/mtcli.py detect
  python tools/mtcli.py list
  python tools/mtcli.py build --preset GPU_LegacyWave1.0.4 --use-preset-selection
  python tools/mtcli.py link --project main-terminal

Notes:
  - Creating a junction under certain folders may require an elevated shell on Windows.
  - Default root is the repository directory that contains this script (.. up to alglib).
"""
import argparse, os, sys, subprocess, hashlib, shutil, datetime, platform, time, json, math, shlex, tempfile
import ctypes
from ctypes import wintypes
from pathlib import Path
from collections import deque

FRIENDLY_SECTIONS = [
    (
        'Essenciais',
        [
            ('detect', 'Localiza Terminals e bibliotecas do projeto'),
            ('list', 'Lista artefatos do build Release'),
            ('link', 'Cria links das pastas MQL5\\Libraries para o Release atual'),
            ('build', 'Configura/compila targets selecionados via CMake'),
            ('build-all', 'Regenera colas/exports e compila Core + Bridge'),
            ('ping', 'Inicializa o mt-bridge para conferir versão/estatísticas'),
            ('env', 'Mostra diagnóstico de ambiente (paths, variáveis, Python)'),
        ],
    ),
    (
        'Código & Testes',
        [
            ('meta', 'Compila um indicador/EA através do MetaEditor'),
            ('badheader', 'Valida diretivas/imports obrigatórios no indicador'),
            ('fft', 'Executa FFTs via mt-bridge para sanity-check da GPU'),
            ('tester', 'Dispara o Strategy Tester com ini já configurado'),
            ('matrix', 'Executa matrizes de teste descritas em YAML'),
            ('package', 'Empacota os binários/resultados para entrega'),
        ],
    ),
    (
        'Automação MT5',
        [
            ('tools/agentctl', 'Wrapper bash que orquestra mtcli + slash-mql5'),
            ('tools/mt_agent_recipes', 'Scripts prontos (install → attach → tester)'),
            ('tools/mt_agent_pack', 'Hooks/Templates para CommandListenerEA'),
        ],
    ),
]


def print_friendly_overview():
    print("MTCLI — Gen2Alglib (projeto‑cêntrico)")
    print("Uso: mtcli <comando> [opções]\n")
    print("Terminal:")
    print("  terminal status [--repair]    Saúde do Terminal/Listener e logs")
    print("  terminal chart indicator attach|detach …      Opera no gráfico")
    print("  terminal screenshot|create|tester|kill        Utilitários do Terminal")
    print() 
    print("Projeto:")
    print("  project init|save|use|show|list; defaults set|show; detect|env")
    print() 
    print("DLL (núcleo GPU/bridge):")
    print("  dll build|build-all|link|fetch-cuda|package|ping|fft|heartbeat")
    print() 
    print("MetaEditor:")
    print("  metaeditor compile --file <.mq5|.ex5> [--syntax-only] [--include]")
    print() 
    print("Utilitários:")
    print("  config lang set|show; env|detect; kill; start (legado)")
    print() 
    print("Dica: use 'mtcli --help' para ver opções detalhadas.\n")


WSL_INTEROP_READY = None


def ensure_wsl_interop() -> bool:
    global WSL_INTEROP_READY
    if WSL_INTEROP_READY is not None:
        return WSL_INTEROP_READY
    if not is_wsl():
        WSL_INTEROP_READY = True
        return True
    try:
        result = subprocess.run(['cmd.exe', '/C', 'ver'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        WSL_INTEROP_READY = (result.returncode == 0)
    except PermissionError:
        WSL_INTEROP_READY = False
    if not WSL_INTEROP_READY:
        print('[interop] cmd.exe não pôde ser executado dentro do WSL.')
        print('[interop] Habilite o Interop adicionando ao /etc/wsl.conf:')
        print('  [interop]\n  enabled=true\n  appendWindowsPath=true')
        print('Depois execute `wsl.exe --shutdown` e abra o WSL novamente.')
    return WSL_INTEROP_READY


# ---------------------------------------------------------------------------
# MT5 interaction helpers (listener/install/chart)
# ---------------------------------------------------------------------------

LOG_SEPARATOR = "=" * 60


def ensure_dir(path: Path | str):
    Path(path).mkdir(parents=True, exist_ok=True)

# ---------------- I18N -----------------
"""
I18N guidelines for contributors:

- Do not print raw user‑facing strings. Use tr('key', **kwargs).
- Add keys for both languages:
    I18N['en']['your_key'] = '…'
    I18N['pt']['your_key'] = '…'
- Keep placeholders explicit (e.g., {path}, {limit}, {rc}).
- The active language is loaded from tools/mtcli_config.json (key 'lang').
- CLI commands:
    mtcli config lang show
    mtcli config lang set --to en|pt
"""
def _config_root():
    try:
        return cli_root_default()
    except Exception:
        return os.path.abspath(os.path.dirname(__file__))

def get_lang() -> str:
    cfg = load_config(_config_root())
    lang = (cfg.get('lang') or 'en').lower()
    return 'pt' if lang.startswith('pt') else 'en'

I18N = {
    'en': {
        'logs_last': "[logs] Last {limit} lines of {tag}: {path}",
        'src_updated': "[bootstrap] Source updated: {path}",
        'src_kept': "[bootstrap] Source kept: {path}",
        'comp_ok': "[bootstrap] Compilation OK: {path}",
        'comp_fail': "[bootstrap] Compilation FAILED (rc={rc}). See {path}",
        'ini_saved': "[listener] INI saved at {path}",
        'cmd_line': "[cmd] {exe} {args}",
        'done_run': "[done] listener run",
        'done': "[done]",
    },
    'pt': {
        'logs_last': "[logs] Últimas {limit} linhas de {tag}: {path}",
        'src_updated': "[bootstrap] Fonte atualizado: {path}",
        'src_kept': "[bootstrap] Fonte mantido: {path}",
        'comp_ok': "[bootstrap] Compilação OK: {path}",
        'comp_fail': "[bootstrap] Compilação FALHOU (rc={rc}). Veja {path}",
        'ini_saved': "[listener] INI salvo em {path}",
        'cmd_line': "[cmd] {exe} {args}",
        'done_run': "[feito] listener run",
        'done': "[feito]",
    }
}

def tr(key: str, **kw) -> str:
    lang = get_lang()
    fmt = I18N.get(lang, I18N['en']).get(key, I18N['en'].get(key, key))
    try:
        return fmt.format(**kw)
    except Exception:
        return fmt


def write_text_utf8(path: Path | str, content: str):
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(content, encoding='utf-8')


def write_text_utf16(path: Path | str, content: str):
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(content, encoding='utf-16-le')


def win_to_wsl(path: str | Path) -> Path:
    s = str(path)
    if not is_wsl():
        return Path(s)
    if s.startswith('/'):  # already wsl style
        return Path(s)
    if len(s) >= 2 and s[1] == ':' and (s[2:3] == '\\' or s[2:3] == '/'):
        drive = s[0].lower()
        rest = s[2:].replace('\\', '/').lstrip('/')
        return Path(f"/mnt/{drive}/{rest}")
    try:
        conv = subprocess.check_output(['wslpath', '-u', s]).decode().strip()
        if conv:
            return Path(conv)
    except Exception:
        pass
    return Path(s)


def wsl_to_win(path: Path) -> str:
    if not is_wsl():
        return str(path)
    try:
        conv = subprocess.check_output(['wslpath', '-w', str(path)]).decode().strip()
        if conv:
            return conv
    except Exception:
        pass
    s = str(path)
    if s.startswith('/mnt/') and len(s) > 6:
        drive = s[5].upper()
        rest = s[7:].replace('/', '\\')
        return f"{drive}:\\{rest}"
    return s


def to_local_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    return win_to_wsl(path)


def to_windows_path(path: Path | str) -> str:
    return wsl_to_win(Path(path))


def run_win_exe(exe: Path | str, args: list[str], detach: bool = False):
    exe_path = Path(exe)
    if is_wsl():
        if not ensure_wsl_interop():
            raise SystemExit('Interop do WSL desabilitado; execute `sudo tee /etc/wsl.conf` com [interop] enabled=true e reinicie o WSL.')
        exe_win = wsl_to_win(exe_path)
        wd_win = wsl_to_win(exe_path.parent)
        if detach:
            ps = 'powershell.exe'
            arglist = ' '.join([f'"{a}"' for a in args])
            ps_cmd = f"Start-Process -FilePath '{exe_win}' -ArgumentList '{arglist}' -WorkingDirectory '{wd_win}'"
            try:
                return subprocess.call([ps, '-NoProfile', '-Command', ps_cmd])
            except FileNotFoundError:
                # Fallback: usar 'start' do cmd.exe (com diretório de trabalho)
                return subprocess.call(['cmd.exe', '/C', 'start', '', '/D', wd_win, exe_win] + args)
        cmd = ['cmd.exe', '/C', exe_win] + args
        return subprocess.call(cmd, cwd=str(exe_path.parent))
    else:
        if detach:
            ps = 'powershell'
            exe_win = str(exe_path)
            arglist = ' '.join([f'"{a}"' for a in args])
            ps_cmd = f"Start-Process -FilePath '{exe_win}' -ArgumentList '{arglist}'"
            try:
                return subprocess.call([ps, '-NoProfile', '-Command', ps_cmd])
            except FileNotFoundError:
                return subprocess.call(['cmd.exe', '/C', 'start', '', exe_win] + args)
        return subprocess.call([str(exe_path)] + args, cwd=str(exe_path.parent))


def timeframe_ok(tf: str) -> str:
    tf = tf.upper()
    allowed = {"M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30",
               "H1","H2","H3","H4","H6","H8","H12","D1","W1","MN1"}
    if tf not in allowed:
        raise SystemExit(f"Timeframe inválido: {tf}")
    return tf


def _ini_escape(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace('\\', '\\\\')


def build_ini_startup(symbol: str | None,
                      period: str | None,
                      template: str | None,
                      expert: str | None,
                      script: str | None,
                      expert_params: str | None,
                      script_params: str | None,
                      shutdown: bool | None) -> str:
    lines = ['[StartUp]']
    if expert:
        # Modo mais compatível: nome do EA sem extensão (MT localizará em Experts\)
        name = expert
        if name.lower().endswith('.ex5'):
            name = name[:-4]
        if name.lower().startswith('experts\\'):
            name = name[len('experts\\'):]
        lines.append(f"Expert={_ini_escape(name)}")
    if script:
        lines.append(f"Script={_ini_escape(script)}")
    if expert_params:
        lines.append(f"ExpertParameters={_ini_escape(expert_params)}")
    if script_params:
        lines.append(f"ScriptParameters={_ini_escape(script_params)}")
    if symbol:
        lines.append(f"Symbol={symbol}")
    if period:
        lines.append(f"Period={period}")
    if template:
        lines.append(f"Template={_ini_escape(template)}")
    if shutdown is not None:
        lines.append(f"ShutdownTerminal={1 if shutdown else 0}")
    return "\n".join(lines) + "\n"


def _mql5_dir_from_data_dir(data_dir: Path) -> Path:
    p = win_to_wsl(data_dir)
    if p.name.upper() != 'MQL5':
        raise SystemExit("Data Folder deve apontar para a pasta 'MQL5'. Atualize o projeto com --data-dir C\\mql5\\MQL5.")
    return p


def _datapath_root_from_data_dir(data_dir: Path) -> Path:
    return _mql5_dir_from_data_dir(data_dir).parent


def _paths_equal(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
    except Exception:
        return str(a).replace('\\','/').lower() == str(b).replace('\\','/').lower()


def _should_use_portable(terminal: Path, data_dir: Path, portable_pref: bool | None) -> tuple[bool, Path]:
    root = _datapath_root_from_data_dir(data_dir)
    term_root = None
    try:
        term_root = win_to_wsl(terminal).parent
    except Exception:
        term_root = None
    if portable_pref is not None:
        want = bool(portable_pref)
        if want and term_root is not None and not _paths_equal(root, term_root):
            # não faz sentido usar portable se o usuário apontou outra Data Folder
            return False, root
        return want, root
    # heurística: se Data Folder == <terminal_dir>/MQL5 → portable
    if term_root is not None and _paths_equal(root, term_root):
        return True, root
    return False, root


def collect_log_targets(data_dir: Path | None) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    if not data_dir:
        return targets
    mql5 = _mql5_dir_from_data_dir(data_dir)
    term_root = mql5.parent
    # Read tail preferences from config (default: terminal only)
    cfg = load_config(cli_root_default())
    want = cfg.get('tail_targets', 'terminal')
    if isinstance(want, str):
        want = [w.strip().lower() for w in want.split(',')]
    else:
        want = [str(x).lower() for x in (want or ['terminal'])]
    include_terminal = 'terminal' in want or 'all' in want
    include_engine = 'engine' in want or 'all' in want
    include_tester = 'tester' in want or 'all' in want
    # Terminal logs (both possible locations)
    if include_terminal:
        targets.append(('terminal', mql5 / 'Logs' / time.strftime('%Y%m%d.log')))
        targets.append(('terminal', term_root / 'Logs' / time.strftime('%Y%m%d.log')))
    # Engine (always under MQL5\Libraries)
    if include_engine:
        targets.append(('engine', mql5 / 'Libraries' / 'logs' / 'gpu_service.log'))
    # Tester logs (both possible locations)
    if include_tester:
        targets.append(('tester', mql5 / 'Tester' / 'logs' / time.strftime('%Y%m%d.log')))
        targets.append(('tester', term_root / 'Tester' / 'logs' / time.strftime('%Y%m%d.log')))
    return targets

def is_listener_active(data_dir: Path) -> bool:
    try:
        send_listener_command(data_dir, 'PING')
        time.sleep(1.2)
        lines = []
        for _, p in collect_log_targets(data_dir):
            lines.extend(tail_lines(p, 60))
        return any('PONG' in (ln or '') for ln in lines)
    except Exception:
        return False

def boot_with_ini_and_schedule(terminal: Path, data_dir: Path, symbol: str, period: str,
                               commands: list[str], portable: bool | None = None, profile: str | None = None,
                               ini_out: Path | None = None) -> int:
    ini = ini_out or (Path.cwd() / 'mt_boot.ini')
    content = build_ini_startup(symbol=symbol,
                                period=timeframe_ok(period),
                                template=None,
                                expert='CommandListenerEA',
                                script=None,
                                expert_params=None,
                                script_params=None,
                                shutdown=False)
    write_text_utf16(ini, content)
    # grava o primeiro comando (o listener lê 1 linha por varredura)
    if commands:
        send_listener_command(data_dir, commands[0])
    extra = [f"/config:{to_windows_path(ini)}"]
    use_portable, dp_root = _should_use_portable(terminal, data_dir, portable)
    # Sempre fornecer um profile explícito
    extra.append(f"/profile:{profile or 'Default'}")
    if use_portable:
        extra.append('/portable')
    else:
        extra.append(f"/datapath:{to_windows_path(dp_root)}")
    rc = run_win_exe(terminal, extra, detach=True)
    time.sleep(1.2)
    print_log_tail('boot with ini', data_dir=data_dir)
    return rc


def tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open('r', encoding='utf-8', errors='replace') as fh:
            lines = [line.rstrip('\r\n') for line in deque(fh, maxlen=limit)]
            # Heurística: se houver muitos NULs, reabrir como UTF-16-LE
            sample = ''.join(lines)[:200]
            if sample.count('\x00') > len(sample) // 4:
                raise UnicodeDecodeError('utf-8', b'', 0, 1, 'suspicious nulls -> try utf-16')
            return lines
    except UnicodeDecodeError:
        with path.open('r', encoding='utf-16', errors='replace') as fh:
            return [line.rstrip('\r\n') for line in deque(fh, maxlen=limit)]


def print_log_tail(tag: str, limit: int = 20, data_dir: Path | None = None):
    try:
        print(LOG_SEPARATOR)
        print(f"[logs] {tag}")
        if not data_dir:
            print('[logs] data_dir não informado.');
            print(LOG_SEPARATOR)
            return
        targets = collect_log_targets(data_dir)
        any_printed = False
        for name, path in targets:
            lines = tail_lines(path, limit)
            if not lines:
                continue
            any_printed = True
            hdr = tr('logs_last', limit=limit, tag=f"{tag} ({name})", path=to_windows_path(path))
            print(hdr)
            for ln in lines:
                print(ln)
            print('-' * 40)
        if not any_printed:
            print('[logs] (sem novas linhas)')
        print(LOG_SEPARATOR)
    except Exception as e:
        print(f"[logs] erro ao ler logs: {e}")
        print(LOG_SEPARATOR)

def print_compile_log_tail(log_path: Path, tag: str = 'meta compile', limit: int = 40):
    try:
        p = Path(log_path)
        if not p.exists():
            return
        print(LOG_SEPARATOR)
        print(f"[logs] Últimas {limit} linhas de {tag}: {to_windows_path(p)}")
        for line in tail_lines(p, limit):
            print(line)
        print(LOG_SEPARATOR)
    except Exception:
        pass


def send_listener_command(data_dir: Path, payload: str) -> Path:
    target_dir = _mql5_dir_from_data_dir(data_dir)
    files_dir = target_dir / 'Files'
    ensure_dir(files_dir)
    cmdfile = files_dir / 'cmd.txt'
    cmdfile.write_text(payload, encoding='ascii')
    return cmdfile


EA_LISTENER_CODE = r'''
#property strict
input string In_CommandFile = "cmd.txt"; // MQL5\Files\cmd.txt

#define ERR_CHART_SUBWINDOW_NOT_FOUND  4109
#define ERR_CHART_INDICATOR_CANNOT_ADD 4114
#define ERR_CHART_INDICATOR_BAD_PARAMETERS 4115

int OnInit(){
   Print("[Listener] DataPath=", TerminalInfoString(TERMINAL_DATA_PATH));
   EventSetTimer(1);
   return(INIT_SUCCEEDED);
}
void OnDeinit(const int _){ EventKillTimer(); }

ENUM_TIMEFRAMES ParseTF(const string s){
   string u = s;
   StringTrimLeft(u);
   StringTrimRight(u);
   StringToUpper(u);
   if(u=="M1") return PERIOD_M1;
   if(u=="M5") return PERIOD_M5;
   if(u=="M15") return PERIOD_M15;
   if(u=="M30") return PERIOD_M30;
   if(u=="H1") return PERIOD_H1;
   if(u=="H4") return PERIOD_H4;
   if(u=="D1") return PERIOD_D1;
   if(u=="W1") return PERIOD_W1;
   if(u=="MN1") return PERIOD_MN1;
   return PERIOD_CURRENT;
}

bool ApplyCommand(const string line){
   string parts[];
   int count = StringSplit(line, ';', parts);
   if(count < 1){ Print("[CommandListener] Linha vazia"); return(false); }
   string cmd = parts[0];
   if(cmd == "ATTACH_IND" && count >= 5){
      string sym = parts[1];
      ENUM_TIMEFRAMES tf = ParseTF(parts[2]);
      string ind = parts[3];
      int sub = (int)StringToInteger(parts[4]);
      long cid = ChartID();
      if(sub > 0){
         cid = ChartOpen(sym, tf);
         if(cid == 0){ Print("ChartOpen falhou: ", GetLastError()); return(false); }
      }
      // Proteção: remover instâncias anteriores com o mesmo nome na mesma subjanela
      int total = ChartIndicatorsTotal(cid, sub);
      for(int i=total-1; i>=0; --i){
         string name = ChartIndicatorName(cid, sub, i);
         if(name == ind){
            if(!ChartIndicatorDelete(cid, sub, name))
               Print("ChartIndicatorDelete (pre-replace) falhou: ", GetLastError());
         }
      }
      int handle = iCustom(sym, tf, ind);
      if(handle == INVALID_HANDLE){ Print("iCustom falhou: ", GetLastError()); return(false); }
      if(!ChartIndicatorAdd(cid, sub, handle)){
         Print("ChartIndicatorAdd falhou: ", GetLastError()); return(false);
      }
      PrintFormat("Indicador %s anexado em %s %s sub=%d", ind, sym, parts[2], sub);
      return(true);
   }
   if(cmd == "DETACH_IND" && count >= 5){
      string sym = parts[1];
      ENUM_TIMEFRAMES tf = ParseTF(parts[2]);
      string ind = parts[3];
      int sub = (int)StringToInteger(parts[4]);
      long cid = ChartID();
      int total = ChartIndicatorsTotal(cid, sub);
      for(int i=total-1; i>=0; --i){
         string name = ChartIndicatorName(cid, sub, i);
         if(name == ind){
            if(!ChartIndicatorDelete(cid, sub, name))
               Print("ChartIndicatorDelete falhou: ", GetLastError());
            else
               PrintFormat("Indicador %s removido de %s %s", ind, sym, parts[2]);
         }
      }
      return(true);
   }
   if(cmd == "ATTACH_EA" && count >= 4){
      string sym = parts[1];
      ENUM_TIMEFRAMES tf = ParseTF(parts[2]);
      string expert = parts[3];
      string tpl = (count >= 5) ? parts[4] : "";
      long chart = ChartOpen(sym, tf);
      if(chart == 0){ Print("ChartOpen falhou: ", GetLastError()); return(false);} 
      if(tpl == ""){
         Print("ATTACH_EA requer um template (.tpl) contendo o EA.");
         return(false);
      }
      if(!ChartApplyTemplate(chart, tpl)){
         Print("ChartApplyTemplate falhou: ", GetLastError());
         return(false);
      }
      ChartRedraw();
      PrintFormat("Expert %s anexado via template %s em %s %s", expert, tpl, sym, parts[2]);
      return(true);
   }
   if(cmd == "DETACH_EA" && count >= 3){
      string sym = parts[1];
      ENUM_TIMEFRAMES tf = ParseTF(parts[2]);
      long chart = ChartOpen(sym, tf);
      if(chart == 0){ Print("ChartOpen falhou: ", GetLastError()); return(false);} 
      // Para remover EA, aplique um template "limpo" (sem EA)
      Print("DETACH_EA: forneça um template sem EA usando APPLY_TPL para limpar o gráfico.");
      return(false);
   }
   if(cmd == "SCREENSHOT" && count >= 6){
      string sym = parts[1];
      ENUM_TIMEFRAMES tf = ParseTF(parts[2]);
      string file = parts[3];
      int w = (int)StringToInteger(parts[4]);
      int h = (int)StringToInteger(parts[5]);
      long chart = ChartOpen(sym, tf);
      if(chart == 0){ Print("ChartOpen falhou: ", GetLastError()); return(false);} 
      if(w <= 0 || h <= 0){
         w = (int)ChartGetInteger(chart, CHART_WIDTH_IN_PIXELS);
         h = (int)ChartGetInteger(chart, CHART_HEIGHT_IN_PIXELS);
      }
      if(!ChartScreenShot(chart, file, w, h)){
         Print("ChartScreenShot falhou: ", GetLastError());
         return(false);
      }
      PrintFormat("Screenshot salvo: %s (%dx%d) em %s %s", file, w, h, sym, parts[2]);
      return(true);
   }
  if(cmd == "APPLY_TPL" && count >= 4){
      string sym = parts[1];
      ENUM_TIMEFRAMES tf = ParseTF(parts[2]);
      string tpl = parts[3];
      long chart = ChartOpen(sym, tf);
      if(chart == 0){ Print("ChartOpen falhou: ", GetLastError()); return(false);} 
      if(!ChartApplyTemplate(chart, tpl))
         Print("ChartApplyTemplate falhou: ", GetLastError());
      else
         PrintFormat("Template %s aplicado em %s %s", tpl, sym, parts[2]);
      return(true);
  }
   if(cmd == "PING"){
      Print("[CommandListener] PONG");
      return(true);
   }
  Print("[CommandListener] Comando desconhecido: ", line);
  return(false);
}

void OnTimer(){
   ResetLastError();
   string path = In_CommandFile;
   if(StringLen(path) == 0)
      path = "cmd.txt";
   string full = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + path;
   if(!FileIsExist(full))
      return;
   int handle = FileOpen(full, FILE_READ|FILE_TXT|FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   string line = FileReadString(handle);
   FileClose(handle);
   FileDelete(full);
   if(StringLen(line) > 0)
      ApplyCommand(line);
}
'''.lstrip()


SCRIPT_APLICAR_TEMPLATE = r'''
#property script_show_inputs
input string In_Symbol = "EURUSD";
input ENUM_TIMEFRAMES In_Timeframe = PERIOD_H1;
input string In_Template = "";

void OnStart(){
   long cid = ChartID();
   string sym = In_Symbol;
   ENUM_TIMEFRAMES tf = In_Timeframe;
   if(!ChartSetSymbolPeriod(cid, sym, tf)){
      Print("ChartSetSymbolPeriod falhou: ", GetLastError());
      return;
   }
   if(In_Template==""){
      Print("Nenhum template informado");
      return;
   }
   if(!ChartApplyTemplate(cid, In_Template))
      Print("Falha ChartApplyTemplate: ", GetLastError());
   else
      PrintFormat("Template '%s' aplicado em %s %s", In_Template, sym, EnumToString(tf));
}
'''.lstrip()


SCRIPT_DUMP_DATAPATH = r'''
#property strict
void OnStart(){
   string dp = TerminalInfoString(TERMINAL_DATA_PATH);
   int h = FileOpen("datapath.txt", FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(h==INVALID_HANDLE){ Print("[Verify] FileOpen failed: ", GetLastError()); return; }
   FileWriteString(h, dp);
   FileClose(h);
   Print("[Verify] DataPath=", dp);
}
'''.lstrip()

def ensure_source(metaeditor: Path | None,
                  data_dir: Path,
                  rel_path: str,
                  code: str,
                  force: bool = False,
                  quiet: bool = False,
                  compile_after: bool = True) -> Path:
    target = _mql5_dir_from_data_dir(data_dir) / rel_path
    ensure_dir(target.parent)
    if force or not target.exists():
        write_text_utf8(target, code)
        if not quiet:
            print(tr('src_updated', path=to_windows_path(target)))
    elif not quiet:
        print(tr('src_kept', path=to_windows_path(target)))

    if compile_after and metaeditor and target.suffix.lower() == '.mq5':
        log = target.with_suffix('.log')
        rc = run_win_exe(metaeditor, [f'/compile:{to_windows_path(target)}', f'/log:{to_windows_path(log)}'])
        # Avalia o resultado pelo log e pela presença do .ex5 (o rc do MetaEditor nem sempre é confiável)
        ok = False
        try:
            lines = tail_lines(log, 80)
            ok = any('Result:' in ln and '0 errors' in ln for ln in lines)
        except Exception:
            ok = False
        if not ok and target.with_suffix('.ex5').exists():
            ok = True
        if ok:
            if not quiet:
                print(tr('comp_ok', path=to_windows_path(target.with_suffix('.ex5'))))
        else:
            print(tr('comp_fail', rc=rc, path=to_windows_path(log)))
        # Sempre retorna um trecho do log para padronizar a saída
        print_compile_log_tail(log, tag='MetaEditor (bootstrap)')
    elif not compile_after and not quiet:
        print(f"[bootstrap] Fonte atualizado sem compilação: {to_windows_path(target)}")
    return target


def bootstrap_instance(metaeditor: Path | None, data_dir: Path, force: bool = False):
    data_path = _mql5_dir_from_data_dir(data_dir)
    ensure_dir(data_path / 'Files')
    ensure_dir(data_path / 'Profiles' / 'Templates')
    ensure_source(metaeditor, data_dir, 'Experts/CommandListenerEA.mq5', EA_LISTENER_CODE, force=force, quiet=False)
    ensure_source(metaeditor, data_dir, 'Scripts/AplicarTemplate.mq5', SCRIPT_APLICAR_TEMPLATE, force=force, quiet=False, compile_after=False)
    ensure_source(metaeditor, data_dir, 'Scripts/DumpDataPath.mq5', SCRIPT_DUMP_DATAPATH, force=force, quiet=False)


def install_file_to_mql(metaeditor: Path | None,
                        data_dir: Path,
                        rel_path: str,
                        src_file: Path,
                        compile_after: bool = True):
    dst = _mql5_dir_from_data_dir(data_dir) / rel_path
    ensure_dir(dst.parent)
    shutil.copy2(src_file, dst)
    print(f"[install] {src_file.name} -> {to_windows_path(dst)}")
    if compile_after and metaeditor and dst.suffix.lower() == '.mq5':
        log = dst.with_suffix('.log')
        rc = run_win_exe(metaeditor, [f'/compile:{to_windows_path(dst)}', f'/log:{to_windows_path(log)}'])
        if rc != 0:
            print(f"[install] Falha na compilação (rc={rc}). Veja {to_windows_path(log)}")
            print_compile_log_tail(log, tag='MetaEditor (install)')
            raise SystemExit(rc)
        print(f"[install] Compilado: {to_windows_path(dst.with_suffix('.ex5'))}")
        print_compile_log_tail(log, tag='MetaEditor (install)')


def ensure_template_present(data_dir: Path, tpl_name: str | None, tpl_source: str | None) -> str | None:
    if not tpl_name and not tpl_source:
        return None
    templates_dir = _mql5_dir_from_data_dir(data_dir) / 'Profiles' / 'Templates'
    ensure_dir(templates_dir)
    if tpl_source:
        src = Path(tpl_source)
        dst = templates_dir / src.name
        shutil.copy2(src, dst)
        print(f"[tpl] {src.name} -> {to_windows_path(dst)}")
        return dst.name
    return tpl_name


def resolve_mt_context(args):
    root = projects_root_from_args(args)
    libs = getattr(args, 'libs', None)
    project = getattr(args, 'project', None)
    pid, libs_default, _ = resolve_project_libs(root, project)
    libs = libs or libs_default
    if not libs:
        raise SystemExit('Projeto não configurado. Defina: mtcli project save --id <nome> --libs <...> --terminal <...> --metaeditor <...> --data-dir <...>')
    libs_path = Path(os.path.abspath(_normalize_path(libs)))
    if not libs_path.exists():
        raise SystemExit(f"Biblioteca não encontrada: {libs_path}")
    # Data Folder deve ser explícita por projeto (sem heurísticas)
    projects = load_projects(root)
    proj_info = projects.get('projects', {}).get(pid, {}) if pid else {}
    data_dir_value = getattr(args, 'data_dir', None) or proj_info.get('data_dir')
    if not data_dir_value:
        raise SystemExit('Data Folder não configurada. Defina: mtcli project save --id <nome> --data-dir <...>')
    data_dir = Path(_normalize_path(data_dir_value))
    cfg = load_config(root)
    terminal = Path(_normalize_path(getattr(args, 'terminal', None) or proj_info.get('terminal') or cfg.get('terminal',''))) if (getattr(args,'terminal',None) or proj_info.get('terminal') or cfg.get('terminal')) else None
    metaeditor = Path(_normalize_path(getattr(args, 'metaeditor', None) or proj_info.get('metaeditor') or cfg.get('metaeditor',''))) if (getattr(args,'metaeditor',None) or proj_info.get('metaeditor') or cfg.get('metaeditor')) else None
    if not terminal:
        raise SystemExit("terminal64.exe não configurado. Vincule o projeto: mtcli project save --id <nome> --libs <...> --terminal <...> --metaeditor <...> --data-dir <...>")
    if not metaeditor:
        print('[warn] metaeditor64.exe não encontrado automaticamente; informe com --metaeditor para compilar indicadores/EAs.')
    return terminal, metaeditor, data_dir


def cmd_listener_install(args):
    terminal, metaeditor, data_dir = resolve_mt_context(args)
    if not metaeditor:
        raise SystemExit('MetaEditor não encontrado; informe --metaeditor.')
    # Instala e compila o listener; se a compilação falhar, retorna erro imediatamente
    bootstrap_instance(metaeditor, data_dir, force=args.force)
    time.sleep(0.2)
    # Verificar o resultado da compilação do Expert
    mql5 = _mql5_dir_from_data_dir(data_dir)
    log = (mql5 / 'Experts' / 'CommandListenerEA.log')
    ex5 = (mql5 / 'Experts' / 'CommandListenerEA.ex5')
    ok = False
    try:
        if log.exists():
            lines = tail_lines(log, 60)
            ok = any('Result:' in ln and '0 errors' in ln for ln in lines)
        if not ok and ex5.exists():
            ok = True
    except Exception:
        ok = ex5.exists()
    print_log_tail('listener install', data_dir=data_dir)
    return 0 if ok else 2


def cmd_listener_run(args):
    # Usa defaults de símbolo/período e portable/profile quando omitidos
    with_project_defaults(args, use_indicator=False)
    terminal, _, data_dir = resolve_mt_context(args)
    if args.indicator:
        line = f"ATTACH_IND;{args.symbol};{timeframe_ok(args.period)};{args.indicator};{args.indicator_subwindow}"
        cmdfile = send_listener_command(data_dir, line)
        print(f"[listener] Comando agendado: {line}")
        print(f"[listener] arquivo: {to_windows_path(cmdfile)}")
    ini = Path(args.ini or (Path.cwd() / 'listener.ini'))
    content = build_ini_startup(
        symbol=args.symbol,
        period=timeframe_ok(args.period),
        template=None,
        expert='CommandListenerEA',
        script=None,
        expert_params=None,
        script_params=None,
        shutdown=False)
    write_text_utf16(ini, content)
    print(tr('ini_saved', path=str(ini)))
    extra = [f"/config:{to_windows_path(ini)}"]
    use_portable, datapath_root = _should_use_portable(terminal, data_dir, getattr(args, 'portable', None))
    # Sempre usar um profile explícito
    prof = getattr(args, 'profile', None) or 'Default'
    extra.append(f"/profile:{prof}")
    if use_portable:
        extra.append('/portable')
    else:
        extra.append(f"/datapath:{to_windows_path(datapath_root)}")
    if getattr(args, 'trace', False):
        try:
            exe_win = to_windows_path(terminal)
            print(tr('cmd_line', exe=exe_win, args=' '.join(extra)))
        except Exception:
            pass
    rc = run_win_exe(terminal, extra, detach=getattr(args, 'detach', True))
    time.sleep(0.5)
    print_log_tail('listener run', data_dir=data_dir)
    print(tr('done_run'))
    return rc


def cmd_listener_ensure(args):
    # Garante que o listener está instalado e inicia o Terminal de forma não-bloqueante
    # Reusa o contexto do projeto atual
    try:
        # sempre reescreve o listener para garantir comandos mais recentes (SCREENSHOT, fixes etc.)
        setattr(args, 'force', True)
        _ = cmd_listener_install(args)
    except SystemExit as e:
        try:
            return int(e.code)
        except Exception:
            print(str(e))
            return 1
    # Se já estiver ativo, não iniciamos outra instância
    def _quick_active_check() -> bool:
        send_listener_command(resolve_mt_context(args)[2], 'PING')
        time.sleep(0.6)
        lines = []
        for _, p in collect_log_targets(resolve_mt_context(args)[2]):
            lines.extend(tail_lines(p, 60))
        return any('PONG' in (ln or '') for ln in lines)

    active = False
    try:
        active = _quick_active_check()
    except Exception:
        active = False
    if active and not getattr(args, 'force_start', False):
        print('[ensure] Listener já ativo — não abrirei nova instância.')
        print_log_tail('listener ensure', data_dir=resolve_mt_context(args)[2])
        return 0
    # Forçar detach como padrão seguro para agentes; garantir atributos esperados
    if not hasattr(args, 'detach'):
        setattr(args, 'detach', True)
    for name, default in [('indicator', None), ('indicator_subwindow', 0), ('symbol','EURUSD'), ('period','H1'), ('ini', None)]:
        if not hasattr(args, name):
            setattr(args, name, default)
    return cmd_listener_run(args)


def _is_up_to_date(src: Path, dst: Path) -> bool:
    try:
        return dst.exists() and src.stat().st_mtime <= dst.stat().st_mtime
    except FileNotFoundError:
        return False


def ensure_indicator_installed(metaeditor: Path | None, data_dir: Path, src_file: Path) -> Path:
    dst = _mql5_dir_from_data_dir(data_dir) / 'Indicators' / src_file.name
    ex5 = dst.with_suffix('.ex5')
    if _is_up_to_date(src_file, ex5):
        print(f"[ensure] Indicador já presente e atualizado: {to_windows_path(ex5)}")
        return ex5
    install_file_to_mql(metaeditor, data_dir, f"Indicators/{src_file.name}", src_file, compile_after=True)
    return ex5


def cmd_indicator_ensure(args):
    with_project_defaults(args, use_indicator=True)
    terminal, metaeditor, data_dir = resolve_mt_context(args)
    if not getattr(args, 'file', None):
        raise SystemExit('Use --file <caminho .mq5|.ex5> para indicar o arquivo a instalar/compilar (o mtcli é agnóstico ao projeto).')
    src = Path(args.file)
    if not src.exists():
        raise SystemExit(f"Arquivo de indicador não encontrado: {src}")
    ex5 = ensure_indicator_installed(metaeditor, data_dir, src)
    # Verifica se o listener está ativo; se não estiver e vamos anexar, garanta listener primeiro
    need_attach = bool(getattr(args, 'attach', False))
    if need_attach:
        def _quick_active_check() -> bool:
            send_listener_command(data_dir, 'PING')
            time.sleep(0.6)
            lines = []
            for _, p in collect_log_targets(data_dir):
                lines.extend(tail_lines(p, 60))
            return any('PONG' in (ln or '') for ln in lines)
        if not _quick_active_check():
            print('[ensure] Listener não respondeu ao PING — iniciando em segundo plano...')
            la = argparse.Namespace(project=getattr(args,'project',None), libs=None, terminal=None, metaeditor=None, data_dir=None, force=False, force_start=False)
            cmd_listener_ensure(la)
            time.sleep(1.0)
    if need_attach:
        a = argparse.Namespace(project=getattr(args, 'project', None), libs=None,
                               terminal=None, metaeditor=None, data_dir=None,
                               symbol=args.symbol, period=args.period,
                               indicator=ex5.stem or args.indicator, subwindow=args.subwindow)
        return chart_indicator_attach(a)
    print_log_tail('indicator ensure', data_dir=data_dir)
    return 0

def cmd_listener_status(args):
    terminal, _, data_dir = resolve_mt_context(args)
    print(f"[ctx] terminal: {to_windows_path(terminal)}")
    print(f"[ctx] data_dir: {to_windows_path(data_dir)}")
    send_listener_command(data_dir, 'PING')
    time.sleep(0.8)
    print_log_tail('listener status', data_dir=data_dir)
    print(tr('done_run'))
    return 0


def cmd_install_indicator(args):
    _, metaeditor, data_dir = resolve_mt_context(args)
    if not data_dir:
        raise SystemExit('Data Folder não encontrada.')
    src = Path(args.file)
    if not src.exists():
        raise SystemExit(f"Arquivo não encontrado: {src}")
    install_file_to_mql(metaeditor, data_dir, f"Indicators/{src.name}", src, compile_after=not args.no_compile)
    time.sleep(0.3)
    print_log_tail('install indicator', data_dir=data_dir)
    print(tr('done'))
    return 0


def cmd_install_expert(args):
    _, metaeditor, data_dir = resolve_mt_context(args)
    if not data_dir:
        raise SystemExit('Data Folder não encontrada.')
    src = Path(args.file)
    if not src.exists():
        raise SystemExit(f"Arquivo não encontrado: {src}")
    install_file_to_mql(metaeditor, data_dir, f"Experts/{src.name}", src, compile_after=not args.no_compile)
    time.sleep(0.3)
    print_log_tail('install expert', data_dir=data_dir)
    print(tr('done'))
    return 0


def chart_indicator_attach(args):
    # Aplicar defaults do projeto quando ausentes
    with_project_defaults(args, use_indicator=True)
    if not getattr(args, 'indicator', None):
        raise SystemExit('Informe --indicator <nome> ou defina um default com `project defaults set --indicator <nome>`.')
    terminal, _, data_dir = resolve_mt_context(args)
    symbol = args.symbol; period = timeframe_ok(args.period); sub = args.subwindow
    line = f"ATTACH_IND;{symbol};{period};{args.indicator};{sub}"
    if is_listener_active(data_dir):
        cmdfile = send_listener_command(data_dir, line)
        print(f"[chart] {line}")
        print(f"[chart] arquivo: {to_windows_path(cmdfile)}")
        time.sleep(0.5)
        print_log_tail('chart indicator attach', data_dir=data_dir)
        print(tr('done'))
        return 0
    # MT fechado → boot com INI + cmd.txt
    print('[dual-mode] listener inativo: inicializando Terminal com INI + comando pendente')
    rc = boot_with_ini_and_schedule(terminal, data_dir, symbol, period, [line], portable=getattr(args, 'portable', None), profile=getattr(args, 'profile', None))
    print(tr('done'))
    return rc


def chart_indicator_detach(args):
    with_project_defaults(args, use_indicator=True)
    if not getattr(args, 'indicator', None):
        raise SystemExit('Informe --indicator <nome> ou defina um default com `project defaults set --indicator <nome>`.')
    terminal, _, data_dir = resolve_mt_context(args)
    symbol = args.symbol; period = timeframe_ok(args.period); sub = args.subwindow
    line = f"DETACH_IND;{symbol};{period};{args.indicator};{sub}"
    if is_listener_active(data_dir):
        cmdfile = send_listener_command(data_dir, line)
        print(f"[chart] {line}")
        print(f"[chart] arquivo: {to_windows_path(cmdfile)}")
        time.sleep(0.5)
        print_log_tail('chart indicator detach', data_dir=data_dir)
        print(tr('done'))
        return 0
    print('[dual-mode] listener inativo: inicializando Terminal com INI + comando pendente')
    rc = boot_with_ini_and_schedule(terminal, data_dir, symbol, period, [line], portable=getattr(args, 'portable', None), profile=getattr(args, 'profile', None))
    print(tr('done'))
    return rc


def chart_expert_attach(args):
    with_project_defaults(args, use_indicator=False)
    terminal, _, data_dir = resolve_mt_context(args)
    tpl_name = ensure_template_present(data_dir, args.template, args.template_src)
    symbol = args.symbol; period = timeframe_ok(args.period)
    line = f"ATTACH_EA;{symbol};{period};{args.expert};{tpl_name or ''}"
    if is_listener_active(data_dir):
        cmdfile = send_listener_command(data_dir, line)
        print(f"[chart] {line}")
        print(f"[chart] arquivo: {to_windows_path(cmdfile)}")
        time.sleep(1.0)
        print_log_tail('chart expert attach', data_dir=data_dir)
        print(tr('done'))
        return 0
    # MT fechado: podemos iniciar diretamente com Expert no INI (resultado equivalente)
    print('[dual-mode] listener inativo: inicializando Terminal com Expert no INI')
    rc = boot_with_ini_and_schedule(terminal, data_dir, symbol, period, commands=[], portable=getattr(args, 'portable', None), profile=getattr(args, 'profile', None))
    print(tr('done'))
    return rc


def chart_expert_detach(args):
    with_project_defaults(args, use_indicator=False)
    terminal, _, data_dir = resolve_mt_context(args)
    symbol = args.symbol; period = timeframe_ok(args.period)
    line = f"DETACH_EA;{symbol};{period}"
    if is_listener_active(data_dir):
        cmdfile = send_listener_command(data_dir, line)
        print(f"[chart] {line}")
        print(f"[chart] arquivo: {to_windows_path(cmdfile)}")
        time.sleep(0.5)
        print_log_tail('chart expert detach', data_dir=data_dir)
        print(tr('done'))
        return 0
    print('[dual-mode] listener inativo: inicializando Terminal com INI + comando pendente')
    rc = boot_with_ini_and_schedule(terminal, data_dir, symbol, period, [line], portable=getattr(args, 'portable', None), profile=getattr(args, 'profile', None))
    print(tr('done'))
    return rc


def chart_template_apply(args):
    # Aplica um template .tpl no gráfico indicado
    with_project_defaults(args, use_indicator=False)
    terminal, _, data_dir = resolve_mt_context(args)
    if not getattr(args, 'template', None):
        raise SystemExit('Informe --template <nome.tpl>')
    symbol = args.symbol; period = timeframe_ok(args.period)
    if is_listener_active(data_dir):
        line = f"APPLY_TPL;{symbol};{period};{args.template}"
        cmdfile = send_listener_command(data_dir, line)
        print(f"[chart] {line}")
        print(f"[chart] arquivo: {to_windows_path(cmdfile)}")
        time.sleep(0.5)
        print_log_tail('chart template apply', data_dir=data_dir)
        print(tr('done'))
        return 0
    # MT fechado → podemos usar Template no INI diretamente (resultado idêntico)
    ini = Path.cwd() / 'mt_boot.ini'
    content = build_ini_startup(symbol=symbol,
                                period=period,
                                template=args.template,
                                expert=None,
                                script=None,
                                expert_params=None,
                                script_params=None,
                                shutdown=False)
    write_text_utf16(ini, content)
    use_portable, dp_root = _should_use_portable(terminal, data_dir, getattr(args, 'portable', None))
    extra = [f"/config:{to_windows_path(ini)}"]
    extra.append(f"/profile:{getattr(args,'profile', None) or 'Default'}")
    if use_portable:
        extra.append('/portable')
    else:
        extra.append(f"/datapath:{to_windows_path(dp_root)}")
    rc = run_win_exe(terminal, extra, detach=True)
    time.sleep(0.8)
    print_log_tail('apply template (boot)', data_dir=data_dir)
    print(tr('done'))
    return rc


def chart_screenshot(args):
    # Gera um screenshot do gráfico indicado. Dual-mode: usa listener se ativo; caso contrário, boot com INI + comando pendente.
    with_project_defaults(args, use_indicator=False)
    terminal, _, data_dir = resolve_mt_context(args)
    symbol = args.symbol
    period = timeframe_ok(args.period)
    width = int(getattr(args, 'width', 0) or 0)
    height = int(getattr(args, 'height', 0) or 0)

    # Arquivo de saída: dentro de MQL5/Files/screenshots por padrão
    out_arg = getattr(args, 'output', None)
    base = _mql5_dir_from_data_dir(data_dir) / 'Files' / 'screenshots'
    ensure_dir(base)
    if out_arg:
        # Se foi passado relativo, salve em MQL5/Files/screenshots
        out_path = Path(out_arg)
        out_path = (base / out_path) if not Path(out_arg).is_absolute() else Path(out_arg)
    else:
        ts = time.strftime('%Y%m%d-%H%M%S')
        out_path = base / f"{symbol}-{period}-{ts}.png"

    # O listener/MQL usa caminhos Windows; converta
    out_win = to_windows_path(out_path)
    line = f"SCREENSHOT;{symbol};{period};{out_win};{width};{height}"

    if is_listener_active(data_dir):
        cmdfile = send_listener_command(data_dir, line)
        print(f"[chart] {line}")
        print(f"[chart] arquivo: {to_windows_path(cmdfile)}")
        time.sleep(0.6)
        print_log_tail('chart screenshot', data_dir=data_dir)
        print(f"[out] {out_win}")
        print(tr('done'))
        return 0

    print('[dual-mode] listener inativo: inicializando Terminal com INI + comando pendente')
    rc = boot_with_ini_and_schedule(terminal, data_dir, symbol, period, [line], portable=getattr(args, 'portable', None), profile=getattr(args, 'profile', None))
    print(f"[out] {out_win}")
    print(tr('done'))
    return rc


def cmd_mt_logs_tail(args):
    terminal, _, data_dir = resolve_mt_context(args)
    print_log_tail('mt logs tail', limit=getattr(args,'lines',40), data_dir=data_dir)
    print(tr('done'))
    return 0

def cmd_gen_mql(args):
    root = cli_root_default()
    gen = os.path.join(root, 'tools', 'bridge_gen', 'alglib_bridge_gen.py')
    if not os.path.isfile(gen):
        print('ERROR: generator not found at', gen)
        return 2
    out_type = args.type or 'indicator'
    out_name = args.name or 'Conexao_ALGLIB'
    cmd = [sys.executable, gen, '--root', root, '--all', '--out-type', out_type, '--out-name', out_name]
    if not getattr(args, 'no_from_exports', False):
        cmd.append('--from-exports')
    if args.yaml:
        cmd += ['--yaml', args.yaml]
    if args.preset:
        cmd += ['--preset', args.preset]
    run(cmd)
    # Resolve source path
    subdir = 'Indicators' if out_type=='indicator' else 'Scripts'
    src_path = os.path.join(root, subdir, f"{out_name}.mq5")
    if not os.path.isfile(src_path):
        print('ERROR: MQL output not found at', src_path)
        return 3
    if not args.copy:
        print('Generated:', src_path)
        return 0
    # Copy to Terminals
    tdirs = terminal_dirs()
    if args.terminal_id:
        tdirs = [d for d in tdirs if d['id']==args.terminal_id]
    if args.mql5_root:
        subdir = 'Indicators' if out_type=='indicator' else 'Scripts'
        dst_dir = os.path.join(args.mql5_root, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        st = os.stat(dst)
        print('\nMQL copied to explicit MQL5 root:')
        print_table([["manual", subdir, os.path.basename(dst), fmt_dt(st.st_mtime), to_winpath(dst) if is_wsl() else dst]],
                    headers=['TerminalID','Folder','File','Modified','Destination'])
        return 0
    if not tdirs:
        print('No Terminal dirs found.')
        return 4
    rows = []
    for d in tdirs if (args.all or args.terminal_id) else terminal_dirs():
        dst_root = d['root']
        dst_dir = os.path.join(dst_root, 'MQL5', subdir)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        st = os.stat(dst)
        rows.append([d['id'], subdir, os.path.basename(dst), fmt_dt(st.st_mtime), to_winpath(dst) if is_wsl() else dst])
    print('\nMQL copied to Terminals:')
    print_table(rows, headers=['TerminalID','Folder','File','Modified','Destination'])
    return 0



def cmd_mt_ini_create(args):
    with_project_defaults(args, use_indicator=False)
    ini = Path(getattr(args, 'output', None) or 'listener.ini')
    content = build_ini_startup(symbol=args.symbol,
                                period=timeframe_ok(args.period),
                                template=getattr(args, 'template', None),
                                expert=getattr(args, 'expert', 'CommandListenerEA'),
                                script=getattr(args, 'script', None),
                                expert_params=getattr(args, 'expert_params', None),
                                script_params=getattr(args, 'script_params', None),
                                shutdown=bool(getattr(args, 'shutdown', False)))
    write_text_utf16(ini, content)
    print(f"[ini] criado: {ini}")
    return 0

def is_windows():
    return platform.system().lower().startswith('win')

def is_wsl():
    try:
        # typical WSL indicator
        return 'microsoft' in platform.uname().release.lower() or 'WSL_DISTRO_NAME' in os.environ
    except Exception:
        return False


def cli_root_default() -> str:
    """Retorna o diretório base do mtcli (estável), ignorando o CWD.
    Pode ser sobrescrito por MTCLI_ROOT (env) ou --root nos subcomandos
    que aceitam esse parâmetro.
    """
    env = os.environ.get('MTCLI_ROOT')
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.dirname(__file__))


def projects_root_from_args(args) -> str:
    r = getattr(args, 'root', None)
    return os.path.abspath(r) if r else cli_root_default()


def _to_windows_path(path: str) -> str:
    if not path:
        return path
    norm = os.path.abspath(path)
    if norm.startswith('/mnt/') and len(norm) > 6:
        drive = norm[5].upper()
        rest = norm[7:].replace('/', '\\')
        return f"{drive}:\\{rest}"
    return norm


def _normalize_path(path: str) -> str:
    if not path:
        return path
    path = path.strip().strip('"')
    if os.name != 'nt' and len(path) >= 2 and path[1] == ':' and path[0].isalpha():
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/').lstrip('/')
        return f"/mnt/{drive}/{rest}"
    return path

def build_dirs(root):
    # Preferred build output dirs (Release), dist first
    dist_env = os.environ.get('WAVE_DIST_ROOT')
    dirs = []
    if dist_env:
        dirs.append(os.path.join(dist_env, 'Release'))
    dirs += [
        os.path.join(root, 'dist-wave', 'Release'),
        os.path.join(root, 'build_bridge', 'Release'),
        os.path.join(root, 'build_service', 'Release'),
        os.path.join(root, 'build_core', 'Release'),
        os.path.join(root, 'build_service-core', 'Release'),
        os.path.join(root, 'build-win', 'Release'),
    ]
    return dirs

def dist_wave_dir(root):
    """Return a directory to preview artifacts from. Prefer dist-wave/Release that contains alglib_bridge.dll.
    Fallback to the first candidate in build_dirs(root).
    """
    candidates = build_dirs(root)
    for d in candidates:
        if os.path.isfile(os.path.join(d, 'alglib_bridge.dll')):
            return d
    return candidates[0] if candidates else root

def default_bridge_path(root):
    if is_windows():
        candidates = [
            os.path.join(root, 'build-win', 'Release', 'mt-bridge.dll'),
            os.path.join(root, 'build-win', 'mt-bridge.dll'),
        ]
    else:
        candidates = [
            os.path.join(root, 'build', 'libmt-bridge.so'),
            os.path.join(root, 'build', 'mt-bridge.dll'),
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def default_build_dir(root):
    if is_windows() or is_wsl():
        return os.path.join(root, 'build-win')
    return os.path.join(root, 'build-wsl')


def _copy_cuda_runtime_dlls(root: str, release_dir: str) -> None:
    cuda_dir = os.path.join(root, 'cuda-files')
    release_abs = os.path.abspath(release_dir)
    if not os.path.isdir(release_abs):
        print(f"[build] Release dir missing, cannot copy CUDA DLLs: {release_abs}")
        return
    if not os.path.isdir(cuda_dir):
        print(f"[build] CUDA runtime directory missing: {cuda_dir}")
        return

    dlls = [name for name in os.listdir(cuda_dir) if name.lower().endswith('.dll')]
    if not dlls:
        print(f"[build] No CUDA DLLs found under {cuda_dir}; skipping copy.")
        return

    os.makedirs(release_abs, exist_ok=True)
    copied = 0
    for name in dlls:
        src = os.path.join(cuda_dir, name)
        dst = os.path.join(release_abs, name)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except OSError as exc:
            print(f"[build] Failed to copy {src} -> {dst}: {exc}")
    dest_display = to_winpath(release_abs) if is_wsl() else release_abs
    print(f"[build] Copied {copied} CUDA runtime DLL(s) into {dest_display}")


def _ensure_link(libs_path: str, release_dir: str) -> tuple[str, str]:
    libs_abs = os.path.abspath(_normalize_path(libs_path))
    release_abs = os.path.abspath(_normalize_path(release_dir))
    libs_win = _to_windows_path(libs_abs)
    release_win = _to_windows_path(release_abs)
    if not os.path.isdir(release_abs):
        return 'ERROR', f'Release dir not found: {release_abs}'
    exists = os.path.exists(libs_abs)
    try:
        if exists and os.path.samefile(libs_abs, release_abs):
            return 'OK', 'already linked'
    except Exception:
        pass
    if exists:
        return 'SKIP', 'path exists (remove manually)'
    os.makedirs(os.path.dirname(libs_abs), exist_ok=True)
    temp_root = os.environ.get('TEMP') or os.environ.get('TMP') or r'C:\Windows\Temp'
    temp_root_norm = _normalize_path(temp_root)
    os.makedirs(temp_root_norm, exist_ok=True)
    fd, script_path = tempfile.mkstemp(prefix='mtcli_link_', suffix='.cmd', dir=temp_root_norm)
    script_win = _to_windows_path(script_path)
    try:
        with os.fdopen(fd, 'w', newline='\r\n') as fh:
            fh.write('@echo off\r\n')
            fh.write(f'mklink /J "{libs_win}" "{release_win}"\r\n')
        cmdline = ['cmd.exe', '/C', script_win]
        print('> ' + ' '.join(cmdline))
        proc = subprocess.run(cmdline, capture_output=True)
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass
    def _dec(data: bytes) -> str:
        if not data:
            return ''
        for enc in ('utf-8', 'cp1252', 'cp850', 'latin-1'):
            try:
                return data.decode(enc).strip()
            except UnicodeDecodeError:
                continue
        return data.decode('latin-1', errors='ignore').strip()
    if proc.returncode == 0:
        out = _dec(proc.stdout)
        if out:
            print(out)
        err = _dec(proc.stderr)
        if err:
            print(err)
        return 'LINKED', release_win
    detail = _dec(proc.stderr) or _dec(proc.stdout) or f'exit {proc.returncode}'
    return 'ERROR', detail

def warn_wsl_path(path, context):
    if not is_wsl() or not path:
        return
    norm = os.path.abspath(path)
    if norm.startswith('/mnt/'):
        print(f"[WARN] {context} located on Windows filesystem ({norm}). If Windows build tools are also using it, clean the directory from Windows or use a WSL-only folder.")

STATUS_TEXT = {
    0: 'OK',
    -1: 'BAD_ARGS',
    -2: 'BACKEND_UNAVAILABLE',
    -3: 'TIMEOUT',
    -4: 'INTERNAL_ERROR',
}

def status_to_string(code: int) -> str:
    return STATUS_TEXT.get(code, f'ERR_{code}')

def _load_bridge_library(dll_path: str, root: str):
    resolved = dll_path or default_bridge_path(root)
    if not resolved:
        raise FileNotFoundError('mt-bridge DLL not found; build it first or pass --dll')
    norm = os.path.normpath(resolved)
    if not os.path.isfile(norm):
        raise FileNotFoundError(f'Bridge DLL missing: {norm}')
    print(f"[mtcli] loading bridge: {norm}")
    try:
        lib = ctypes.WinDLL(norm) if is_windows() else ctypes.cdll.LoadLibrary(norm)
    except OSError as exc:
        raise RuntimeError(f'Failed to load {norm}: {exc}')
    return lib, norm

def _setup_bridge_function(lib, name, restype, argtypes):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        raise RuntimeError(f'Bridge missing symbol {name}')
    fn.restype = restype
    fn.argtypes = argtypes
    return fn

def get_pipe_name():
    return os.environ.get('WAVE_PIPE', r'\\.\\pipe\\alglib-wave_pipe')

def try_service_shutdown_via_pipe(timeout_ms: int = 1000) -> bool:
    """Attempt a graceful shutdown by sending CONTROL_SHUTDOWN over the Named Pipe."""
    if not is_windows():
        return False
    pipe = get_pipe_name()
    try:
        WaitNamedPipeW = ctypes.windll.kernel32.WaitNamedPipeW
        WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
        WaitNamedPipeW.restype = wintypes.BOOL
        CreateFileW = ctypes.windll.kernel32.CreateFileW
        CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        CreateFileW.restype = wintypes.HANDLE
        WriteFile = ctypes.windll.kernel32.WriteFile
        WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        WriteFile.restype = wintypes.BOOL
        CloseHandle = ctypes.windll.kernel32.CloseHandle
        CloseHandle.argtypes = [wintypes.HANDLE]
        CloseHandle.restype = wintypes.BOOL

        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        OPEN_EXISTING = 3
        INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

        if not WaitNamedPipeW(pipe, timeout_ms):
            return False
        h = CreateFileW(pipe, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, None)
        if h == INVALID_HANDLE_VALUE:
            return False
        import struct, time as _time
        MESSAGE_MAGIC = 0x4C574650
        PROTOCOL_VERSION = 1
        CMD_PROCESS_SUBMIT = 20
        OP_CONTROL_SHUTDOWN = 98
        tag = int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF
        # Little-endian pack: IIIQIIiiI
        header = struct.pack('<IIIQIIiiI',
                             MESSAGE_MAGIC,
                             PROTOCOL_VERSION,
                             CMD_PROCESS_SUBMIT,
                             tag,
                             OP_CONTROL_SHUTDOWN,
                             0,  # flags
                             0,  # window_len
                             0,  # aux_len
                             0)  # param_size
        n = wintypes.DWORD(0)
        ok = WriteFile(h, header, len(header), ctypes.byref(n), None)
        CloseHandle(h)
        return bool(ok)
    except Exception:
        return False


def terminal_dirs():
    dirs = []
    # Portable installs: prefer /mnt/mql5 when available, then /mnt/c/mql5 or C:/mql5
    portable_candidates = []
    if is_wsl():
        portable_candidates += ['/mnt/mql5', '/mnt/c/mql5']
    elif is_windows():
        portable_candidates += ['C:/mql5']
    # Env override has precedence
    env_port = os.environ.get('WAVE_PORTABLE_ROOT')
    if env_port:
        portable_candidates = [env_port] + portable_candidates
    seen = set()
    for root in portable_candidates:
        libs = os.path.join(root, 'MQL5', 'Libraries')
        if os.path.isdir(libs) and libs not in seen:
            dirs.append({'id': 'portable', 'root': root, 'libs': libs})
            seen.add(libs)

    if is_windows():
        appdata = os.environ.get('APPDATA')
        if appdata:
            base = os.path.join(appdata, 'MetaQuotes', 'Terminal')
            if os.path.isdir(base):
                for name in os.listdir(base):
                    p = os.path.join(base, name)
                    if os.path.isdir(p):
                        libs = os.path.join(p, 'MQL5', 'Libraries')
                        if os.path.isdir(libs):
                            dirs.append({'id': name, 'root': p, 'libs': libs})
    elif is_wsl():
        base_users = '/mnt/c/Users'
        if os.path.isdir(base_users):
            for user in os.listdir(base_users):
                tbase = os.path.join(base_users, user, 'AppData', 'Roaming', 'MetaQuotes', 'Terminal')
                if not os.path.isdir(tbase):
                    continue
                for name in os.listdir(tbase):
                    p = os.path.join(tbase, name)
                    if os.path.isdir(p):
                        libs = os.path.join(p, 'MQL5', 'Libraries')
                        if os.path.isdir(libs):
                            dirs.append({'id': name, 'root': p, 'libs': libs})
    return dirs


def find_agent_libs():
    libs = []
    candidates = []
    if is_windows():
        candidates.append(os.path.join('C:\\mql5', 'Tester'))
    if is_wsl():
        candidates.append('/mnt/c/mql5/Tester')
    for root in candidates:
        if not os.path.isdir(root):
            continue
        try:
            for name in os.listdir(root):
                if not name.lower().startswith('agent-'):
                    continue
                # Inclui mesmo se a pasta Libraries ainda não existe; criamos depois.
                p = os.path.join(root, name, 'MQL5', 'Libraries')
                libs.append(p)
        except Exception:
            pass
    # de-dup normalized
    out = []
    seen = set()
    for p in libs:
        np = os.path.normpath(p)
        if np not in seen:
            seen.add(np)
            out.append(np)
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# --- Windows helpers for in-use replacement ---
MOVEFILE_REPLACE_EXISTING = 0x1
MOVEFILE_COPY_ALLOWED = 0x2
MOVEFILE_DELAY_UNTIL_REBOOT = 0x4

def schedule_move_on_reboot(src_tmp: str, dst_final: str) -> bool:
    """Schedule a replace of dst_final by src_tmp at next reboot (Windows only)."""
    if not is_windows():
        return False
    try:
        MoveFileExW = ctypes.windll.kernel32.MoveFileExW
        MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        MoveFileExW.restype = wintypes.BOOL
        flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED | MOVEFILE_DELAY_UNTIL_REBOOT
        ok = bool(MoveFileExW(src_tmp, dst_final, flags))
        return ok
    except Exception:
        return False


def fmt_dt(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def check_admin_windows():
    if not is_windows():
        return True
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def run(cmd, cwd=None, check=True):
    print('> ' + ' '.join(cmd))
    # Do NOT capture output: we want all system returns printed to the console
    return subprocess.run(cmd, cwd=cwd, check=check)


def to_winpath(p: str) -> str:
    r"""Convert /mnt/c/... to C:\... and '/' to '\\' (WSL → Windows)."""
    if not is_wsl() or not isinstance(p, str):
        return p
    if p.startswith('/mnt/') and len(p) > 6 and p[6] == '/':
        drive = p[5].upper()
        rest = p[7:].replace('/', '\\')
        return f"{drive}:\\{rest}"
    return p


# ---------------- Project persistence -----------------
def _projects_path(root: str) -> str:
    return os.path.join(root, 'tools', 'mtcli_projects.json')

def _config_path(root: str) -> str:
    return os.path.join(root, 'tools', 'mtcli_config.json')

def load_config(root: str):
    path = _config_path(root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(root: str, data):
    path = _config_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_projects(root: str):
    path = _projects_path(root)
    if not os.path.isfile(path):
        return {"last_project": None, "projects": {}}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"last_project": None, "projects": {}}

def save_projects(root: str, data):
    path = _projects_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def resolve_project_libs(root: str, project_id: str | None):
    data = load_projects(root)
    pid = project_id or data.get('last_project')
    if not pid:
        return None, None, "Nenhum projeto salvo. Rode 'mtcli project save --id <nome> --libs C:\\caminho\\MQL5\\Libraries'."
    proj = data.get('projects', {}).get(pid)
    if not proj:
        return pid, None, f"Projeto '{pid}' não encontrado em tools/mtcli_projects.json."
    libs = proj.get('libs')
    if not libs:
        return pid, None, f"Projeto '{pid}' não possui caminho 'libs' configurado. Atualize com 'mtcli project save --id {pid} --libs C:\\caminho\\MQL5\\Libraries'."
    return pid, libs, None


def resolve_project_defaults(root: str, project_id: str | None):
    """Retorna dict com defaults do projeto (symbol, period, subwindow, indicator)."""
    data = load_projects(root)
    pid = project_id or data.get('last_project')
    proj = data.get('projects', {}).get(pid, {}) if pid else {}
    defaults = proj.get('defaults', {})
    return {
        'symbol': defaults.get('symbol', 'EURUSD'),
        'period': defaults.get('period', 'H1'),
        'subwindow': int(defaults.get('subwindow', 1)),
        'indicator': defaults.get('indicator') or None,
        'portable': defaults.get('portable'),
        'profile': defaults.get('profile'),
    }


def with_project_defaults(args, use_indicator: bool = True):
    """Preenche args com defaults do projeto quando ausentes.
    Não altera valores já fornecidos na linha de comando.
    """
    root = projects_root_from_args(args)
    defs = resolve_project_defaults(root, getattr(args, 'project', None))
    if not getattr(args, 'symbol', None):
        setattr(args, 'symbol', defs['symbol'])
    if not getattr(args, 'period', None):
        setattr(args, 'period', defs['period'])
    if not getattr(args, 'subwindow', None):
        setattr(args, 'subwindow', defs['subwindow'])
    if use_indicator and not getattr(args, 'indicator', None) and defs['indicator']:
        setattr(args, 'indicator', defs['indicator'])
    # Propagar portable/profile quando presente
    if not hasattr(args, 'portable') or getattr(args, 'portable') is None:
        setattr(args, 'portable', defs.get('portable'))
    if not hasattr(args, 'profile') or getattr(args, 'profile') is None:
        if defs.get('profile'):
            setattr(args, 'profile', defs.get('profile'))
    return args


def cmd_detect(args):
    root = projects_root_from_args(args)
    if args.all:
        tdirs = terminal_dirs()
        if not tdirs:
            print('No Terminal directories found under %APPDATA%/MetaQuotes/Terminal')
            return 0
        print('Detected Terminal directories:')
        for d in tdirs:
            libs_disp = to_winpath(d['libs']) if is_wsl() else d['libs']
            print(f"- id={d['id']} libs={libs_disp}")
        return 0

    pid, libs, err = resolve_project_libs(root, getattr(args, 'project', None))
    if not libs:
        print('ERROR:', err)
        print('Vinculação rígida por projeto: salve todos os caminhos explícitos:')
        print('  mtcli project save --id <nome> \\\n+    --libs C\\caminho\\MQL5\\Libraries \\\n+    --terminal C\\caminho\\terminal64.exe \\\n+    --metaeditor C\\caminho\\metaeditor64.exe \\\n+    --data-dir C\\caminho\\DataFolder')
        print('Dica: rode `mtcli start` para um assistente interativo, ou `mtcli detect --all` para listar Terminals.')
        return 2
    libs_disp = to_winpath(libs) if is_wsl() else libs
    print('Project Libraries:')
    print(f"- project={pid} libs={libs_disp}")
    return 0


def list_file(path):
    if not os.path.isfile(path):
        disp = to_winpath(path) if is_wsl() else path
        print(f'MISSING: {disp}')
        return
    st = os.stat(path)
    age_min = int((datetime.datetime.now().timestamp() - st.st_mtime) / 60)
    disp = to_winpath(path) if is_wsl() else path
    print(f"{disp:80s} | {st.st_size:10d} B | {fmt_dt(st.st_mtime)} | age ~{age_min:3d}m | {sha256_of(path)[:16]}")


def cmd_list(args):
    # Only search the configured build Release directories; do NOT fall back to repo root.
    # This avoids confusing "MISSING: C:\\...\\alglib_core.dll" at the root when the real one is in build_service-core/Release.
    root = projects_root_from_args(args)
    bases = ['alglib_bridge.dll','alglib_core.dll','alglib_worker.exe']
    print('Artifacts:')
    for base in bases:
        candidates = [os.path.join(b, base) for b in build_dirs(root)]
        found = next((c for c in candidates if os.path.isfile(c)), None)
        if found:
            list_file(found)
        else:
            searched = ', '.join(to_winpath(p) if is_wsl() else p for p in candidates)
            print(f"MISSING: {base} (searched: {searched})")
    return 0


def cmd_list_v2(args):
    """Improved list: search only build Release dirs and show where we looked."""
    root = cli_root_default()
    bases = ['alglib_bridge.dll','alglib_core.dll','alglib_worker.exe']
    print('Artifacts:')
    for base in bases:
        candidates = [os.path.join(b, base) for b in build_dirs(root)]
        found = next((c for c in candidates if os.path.isfile(c)), None)
        if found:
            list_file(found)
        else:
            searched = ', '.join(to_winpath(p) if is_wsl() else p for p in candidates)
            print(f"MISSING: {base} (searched: {searched})")
    return 0

def cmd_env(args):
    root = cli_root_default()
    print(f"root: {root}")
    print(f"platform: {'Windows' if is_windows() else 'WSL' if is_wsl() else 'Linux'}")
    print(f"python: {sys.version.split()[0]}")
    build_dir = default_build_dir(root)
    print(f"default build dir: {build_dir}")
    warn_wsl_path(build_dir, 'Default build dir')
    # Report cmake version if available
    try:
        r = subprocess.run(['cmake', '--version'], capture_output=True, text=True, check=False)
        cmake_ver = (r.stdout or r.stderr or '').strip().splitlines()[0]
        print(f"cmake: {cmake_ver}")
    except Exception as exc:
        print(f"cmake: not available ({exc})")
    bridge = default_bridge_path(root)
    if bridge:
        print(f"mt-bridge: {bridge}")
    else:
        print('mt-bridge: not built yet')
    return 0

def cmd_ping(args):
    root = cli_root_default()
    try:
        lib, path = _load_bridge_library(args.dll, root)
    except Exception as exc:
        print(f"[ERR] {exc}")
        return 2
    gpu_init = _setup_bridge_function(lib, 'gpu_init', ctypes.c_int, [ctypes.c_int, ctypes.c_int])
    gpu_shutdown = _setup_bridge_function(lib, 'gpu_shutdown', None, [])
    gpu_get_version = getattr(lib, 'gpu_get_version', None)
    status = gpu_init(args.device, args.streams)
    if status != 0:
        print(f"gpu_init failed: {status_to_string(status)} ({status})")
        return status
    version = None
    if gpu_get_version:
        gpu_get_version.argtypes = [ctypes.c_char_p, ctypes.c_int]
        gpu_get_version.restype = ctypes.c_int
        buf = ctypes.create_string_buffer(128)
        if gpu_get_version(buf, ctypes.sizeof(buf)) == 0:
            version = buf.value.decode('utf-8', errors='ignore')
    print(f"bridge={path} init_status={status_to_string(status)} version={version or 'n/a'}")
    stats_fn = getattr(lib, 'gpu_get_stats', None)
    if stats_fn:
        stats_fn.argtypes = [ctypes.POINTER(_AlglibCoreStats)]
        stats_fn.restype = ctypes.c_int
        stats = _AlglibCoreStats()
        st = stats_fn(ctypes.byref(stats))
        if st == 0:
            last_update = _format_last_update(stats)
            last_status = getattr(stats, 'last_status', 0)
            print(
                f"backend={_backend_name(stats.backend_type)} inflight={stats.inflight_jobs} "
                f"queue={stats.queue_depth}/{stats.queue_limit} avg_ms={getattr(stats, 'avg_job_duration_ms', 0)} "
                f"last_ms={getattr(stats, 'last_job_duration_ms', 0)} status={status_to_string(last_status)}({last_status}) "
                f"updated={last_update}"
            )
        else:
            print(f"gpu_get_stats failed: {status_to_string(st)}")
    gpu_shutdown()
    return 0

def cmd_config(args):
    root = cli_root_default()
    current = load_config(root)
    payload = getattr(args, 'payload', None)
    if not payload:
        print(json.dumps(current, indent=2) if current else '{}')
        return 0
    try:
        payload = json.loads(payload)
    except json.JSONDecodeError as exc:
        print('Invalid JSON:', exc)
        return 2
    if not isinstance(payload, dict):
        print('Config payload must be a JSON object')
        return 2
    current.update(payload)
    save_config(root, current)
    print('Config updated:')
    print(json.dumps(current, indent=2))
    return 0

def cmd_fft(args):
    root = cli_root_default()
    try:
        lib, _ = _load_bridge_library(args.dll, root)
    except Exception as exc:
        print(f"[ERR] {exc}")
        return 2
    gpu_init = _setup_bridge_function(lib, 'gpu_init', ctypes.c_int, [ctypes.c_int, ctypes.c_int])
    gpu_shutdown = _setup_bridge_function(lib, 'gpu_shutdown', None, [])
    fft_real = _setup_bridge_function(lib, 'gpu_fft_real_forward', ctypes.c_int, [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)])
    status = gpu_init(args.device, args.streams)
    if status != 0:
        print(f"gpu_init failed: {status_to_string(status)} ({status})")
        return status
    try:
        n = args.size
        src_type = ctypes.c_double * n
        dst_type = ctypes.c_double * n
        import time
        start = time.time()
        for rep in range(args.repeat):
            inp = src_type()
            for i in range(n):
                inp[i] = math.sin(2 * math.pi * i / n) + 0.5 * math.cos(4 * math.pi * i / n)
            out = dst_type()
            st = fft_real(inp, n, out)
            if st != 0:
                print(f"fft_real_forward failed on iteration {rep}: {status_to_string(st)}")
                return st
        elapsed = time.time() - start
        print(f"FFT {n} points x{args.repeat} completed in {elapsed:.3f}s ({elapsed/args.repeat if args.repeat else 0:.4f}s per call)")
    finally:
        gpu_shutdown()
    return 0

def cmd_badheader(args):
    root = cli_root_default()
    indicator = args.indicator or os.path.join(root, 'Indicators', 'Conexao_ALGLIB.mq5')
    if not os.path.isfile(indicator):
        print('Indicator not found:', indicator)
        return 2
    with open(indicator, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [next(f) for _ in range(40)]
    checks = {
        '#property strict': False,
        '#property indicator_separate_window': False,
        '#import "mt-bridge.dll"': False,
    }
    for line in lines:
        for key in checks:
            if key in line:
                checks[key] = True
    ok = True
    for key, present in checks.items():
        print(f"{key}: {'OK' if present else 'MISSING'}")
        ok &= present
    return 0 if ok else 1

def cmd_meta(args):
    indicator = args.indicator
    if not indicator or not os.path.isfile(indicator):
        print('Indicator path required (--indicator)')
        return 2
    meta = args.metaeditor or os.environ.get('METAEDITOR64') or os.environ.get('METAEDITOR')
    if not meta:
        default = 'C\\mql5\\MetaEditor64.exe'
        meta = args.metaeditor or default
    meta = to_winpath(meta) if is_wsl() else meta
    log_path = args.log or os.path.join(os.path.dirname(indicator), 'meta_compile.log')
    cmd = [meta, '/compile:', indicator, '/log:', log_path]
    print('Running MetaEditor compile...')
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print('MetaEditor not found at', meta)
        return 2
    try:
        print_compile_log_tail(Path(log_path), tag='MetaEditor (meta)')
    except Exception:
        pass
    return proc.returncode


# ---- fetch-cuda (top-level function for reuse in dll group and legacy) ----
def cmd_fetch_cuda(args):
    import subprocess, pathlib
    root = cli_root_default()
    temp = pathlib.Path(root) / 'cuda_redist_temp'
    temp.mkdir(parents=True, exist_ok=True)
    cmds = [
        'curl -L -o cuda_runtime.zip https://developer.download.nvidia.com/compute/cuda/redist/cuda_runtime/windows/x86_64/cuda_runtime-windows-13.0.0.0.zip',
        'curl -L -o cufft.zip https://developer.download.nvidia.com/compute/cuda/redist/cufft/windows/x86_64/cufft-windows-13.0.0.0.zip',
        'curl -L -o cublas.zip https://developer.download.nvidia.com/compute/cuda/redist/cublas/windows/x86_64/cublas-windows-13.0.0.0.zip',
        'curl -L -o cublas_lt.zip https://developer.download.nvidia.com/compute/cuda/redist/cublas_lt/windows/x86_64/cublas_lt-windows-13.0.0.0.zip',
        'curl -L -o nvrtc.zip https://developer.download.nvidia.com/compute/cuda/redist/nvrtc/windows/x86_64/nvrtc-windows-13.4.0.0.zip',
    ]
    for cmd in cmds:
        print('>', cmd)
        subprocess.check_call(cmd, shell=True, cwd=temp)
    print('Downloads salvos em', temp)
    print('Extraia cada ZIP e copie os DLLs (cudart64_130.dll, cufft64_13.dll, cublas64_13.dll, cublasLt64_13.dll, nvrtc64_134_0.dll) para C:/mql5/MQL5/Libraries.')

def cmd_tester(args):
    root = cli_root_default()
    script = os.path.join(root, 'tools', 'tester', 'run_fft_suite.py')
    if not os.path.isfile(script):
        print('Tester script not found:', script)
        return 2
    cmd = [sys.executable, script]
    if args.config:
        cmd += ['--config', args.config]
    return run(cmd).returncode

def cmd_matrix(args):
    root = cli_root_default()
    workflow = args.workflow or os.path.join(root, '.github', 'workflows', 'matrix-ci.yml')
    if os.path.isfile(workflow):
        print('Matrix workflow:', workflow)
        with open(workflow, 'r', encoding='utf-8', errors='ignore') as f:
            snippet = ''.join(f.readlines()[:20])
        print('--- workflow head ---')
        print(snippet)
        print('--- end head ---')
        return 0
    print('Workflow not found:', workflow)
    return 2

def cmd_package(args):
    root = cli_root_default()
    script = os.path.join(root, 'packaging', 'release.ps1')
    if not os.path.isfile(script):
        print('Packaging script not found:', script)
        return 2
    out_dir = args.output or os.path.join(root, 'dist', 'releases')
    os.makedirs(out_dir, exist_ok=True)
    ps_cmd = [
        'powershell.exe' if is_wsl() else 'powershell',
        '-NoProfile',
        '-Command',
        f"& '{script}' -Version '{args.version}' -Output '{out_dir}'"
    ]
    return run(ps_cmd).returncode

def cmd_build(args):
    root = cli_root_default()
    build = args.build_dir or default_build_dir(root)
    arch = args.arch or 'x64'
    config = args.config or 'Release'

    print(f"[build] root={root}")
    print(f"[build] build_dir={build} config={config}")
    use_windows_toolchain = (is_windows() or is_wsl()) and not args.no_vcvars
    if not use_windows_toolchain:
        warn_wsl_path(build, 'Build directory')
    elif is_wsl():
        print('[build] WSL detected — invoking Windows toolchain via vcvarsall for build-win output.')

    # Optional preset regeneration
    if args.preset:
        opgen = os.path.join(root, 'tools', 'bridge_gen', 'alglib_bridge_gen.py')
        if not os.path.isfile(opgen):
            print('ERROR: opgen.py not found at', opgen)
            return 2
        regen_cmd = [sys.executable, opgen, '--root', root, '--preset', args.preset]
        if args.use_preset_selection:
            regen_cmd.append('--use-preset-selection')
        if args.no_union_presets:
            regen_cmd.append('--no-union-presets')
        run(regen_cmd)

    if use_windows_toolchain:
        vcvars = args.vcvars or r"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat"
        vcvars_check = _normalize_path(vcvars) if is_wsl() else vcvars
        if not os.path.isfile(vcvars_check):
            print(f"ERROR: vcvarsall not found at {vcvars}")
            return 3
        win_root = _to_windows_path(root)
        win_build = _to_windows_path(build)
        generator = args.generator or 'Visual Studio 17 2022'
        toolset = args.toolset or 'host=x64'
        cmake_cfg = [
            'cmake', '-S', win_root, '-B', win_build,
            '-G', generator,
            '-T', toolset,
            '-A', arch,
        ]
        cfg_cmd = subprocess.list2cmdline(cmake_cfg)
        build_cmd_list = ['cmake', '--build', win_build, '--config', config]
        if args.targets:
            build_cmd_list += ['--target'] + args.targets
        if args.parallel:
            build_cmd_list += ['-j', str(args.parallel)]
        if args.extra:
            build_cmd_list += args.extra
        build_str = subprocess.list2cmdline(build_cmd_list)
        full_cmd = f'call "{vcvars}" x64 && cd /d "{win_root}" && {cfg_cmd} && {build_str}'
        if is_wsl():
            temp_root = os.environ.get('TEMP') or os.environ.get('TMP') or r'C:\Windows\Temp'
            temp_root_norm = _normalize_path(temp_root)
            os.makedirs(temp_root_norm, exist_ok=True)
            fd, script_path = tempfile.mkstemp(prefix='mtcli_build_', suffix='.cmd', dir=temp_root_norm)
            script_win = _to_windows_path(script_path)
            try:
                with os.fdopen(fd, 'w', newline='\r\n') as fh:
                    fh.write('@echo off\r\n')
                    fh.write(full_cmd + '\r\n')
                run(['cmd.exe', '/C', script_win])
            finally:
                try:
                    os.remove(script_path)
                except OSError:
                    pass
        else:
            run(['cmd.exe', '/C', full_cmd])
    else:
        cfg_cmd = ['cmake', '-S', root, '-B', build]
        if is_windows():
            cfg_cmd += ['-A', arch]
        if args.generator:
            cfg_cmd += ['-G', args.generator]
        if args.toolset:
            cfg_cmd += ['-T', args.toolset]
        run(cfg_cmd)

        build_cmd = ['cmake', '--build', build, '--config', config]
        if args.targets:
            build_cmd += ['--target'] + args.targets
        if args.parallel:
            build_cmd += ['-j', str(args.parallel)]
        if args.extra:
            build_cmd += args.extra
        run(build_cmd)

    release_dir = os.path.join(build, config)
    _copy_cuda_runtime_dlls(root, release_dir)
    return 0


def cmd_build_all(args):
    root = cli_root_default()
    arch = args.arch or 'x64'
    config = args.config or 'Release'
    gen = os.path.join(root, 'tools', 'bridge_gen', 'alglib_bridge_gen.py')

    print(f"[build-all] root={root} arch={arch} config={config}")

    # 0) Generate glue+exports (always), optional preset for indicator stub
    if not os.path.isfile(gen):
        print('ERROR: generator not found at', gen)
        return 2
    gen_cmd = [sys.executable, gen, '--root', root, '--all']
    if args.preset:
        gen_cmd += ['--preset', args.preset]
    run(gen_cmd)

    # 1) Configure + build Core/Service (single build dir)
    b1 = os.path.join(root, 'build_core')
    warn_wsl_path(b1, 'build_core')
    cfg1 = ['cmake', '-S', root, '-B', b1]
    if is_windows():
        cfg1 += ['-A', arch]
    run(cfg1)

    run(['cmake', '--build', b1, '--config', config, '--target', 'alglib_core', '-j', str(args.parallel or 8)])
    run(['cmake', '--build', b1, '--config', config, '--target', 'alglib_worker', '-j', str(args.parallel or 8)])

    # 2) Configure + build Bridge
    b2 = os.path.join(root, 'build_bridge')
    warn_wsl_path(b2, 'build_bridge')
    cfg2 = ['cmake', '-S', root, '-B', b2]
    if is_windows():
        cfg2 += ['-A', arch]
    run(cfg2)

    run(['cmake', '--build', b2, '--config', config, '--target', 'alglib_bridge', '-j', str(args.parallel or 8)])

    # 3) Summary
    dw = dist_wave_dir(root)
    print('\nBuild outputs:')
    for fn in ['alglib_core.dll','alglib_worker.exe','alglib_bridge.dll']:
        list_file(os.path.join(dw, fn))
    return 0


class _AlglibCoreStats(ctypes.Structure):
    _fields_ = [
        ('total_jobs_submitted', ctypes.c_uint64),
        ('total_jobs_completed', ctypes.c_uint64),
        ('avg_job_duration_ms', ctypes.c_uint64),
        ('last_job_duration_ms', ctypes.c_uint64),
        ('last_update_epoch_ms', ctypes.c_uint64),
        ('inflight_jobs', ctypes.c_int32),
        ('worker_threads', ctypes.c_int32),
        ('queue_depth', ctypes.c_int32),
        ('queue_limit', ctypes.c_int32),
        ('backend_type', ctypes.c_int32),
        ('last_status', ctypes.c_int32),
        ('reserved0', ctypes.c_int32),
        ('reserved1', ctypes.c_int32),
    ]


def _backend_name(backend_id: int) -> str:
    mapping = {
        0: 'AUTO',
        1: 'CPU',
        2: 'CUDA',
        3: 'OPENCL',
    }
    return mapping.get(backend_id, f'UNKNOWN({backend_id})')


def _format_last_update(stats: _AlglibCoreStats) -> str:
    try:
        epoch = getattr(stats, 'last_update_epoch_ms', 0)
        if epoch and epoch > 0:
            dt = datetime.datetime.utcfromtimestamp(epoch / 1000.0)
            return dt.strftime('%Y-%m-%d %H:%M:%S') + ' UTC'
    except Exception:
        pass
    return 'n/a'


def cmd_heartbeat(args):
    root = cli_root_default()
    dll_path = args.dll or default_bridge_path(root)
    if not dll_path:
        print('[ERR] Bridge DLL not found; specify --dll or build artifacts first')
        return 2
    if not os.path.isfile(dll_path):
        print(f"[ERR] Bridge DLL not found: {dll_path}")
        return 2

    try:
        lib = ctypes.WinDLL(dll_path) if is_windows() else ctypes.cdll.LoadLibrary(dll_path)
    except OSError as exc:
        print(f"[ERR] Failed to load {dll_path}: {exc}")
        return 2

    try:
        gpu_get_stats = lib.gpu_get_stats
    except AttributeError:
        print('[ERR] gpu_get_stats not exported by bridge DLL')
        return 2

    gpu_get_stats.argtypes = [ctypes.POINTER(_AlglibCoreStats)]
    gpu_get_stats.restype = ctypes.c_int

    gpu_get_version = getattr(lib, 'gpu_get_version', None)
    version_label = None
    if gpu_get_version:
        gpu_get_version.argtypes = [ctypes.c_char_p, ctypes.c_int]
        gpu_get_version.restype = ctypes.c_int
        buf = ctypes.create_string_buffer(128)
        if gpu_get_version(buf, ctypes.sizeof(buf)) == 0:
            try:
                version_label = buf.value.decode('utf-8', errors='ignore')
            except Exception:
                version_label = buf.value.decode('latin-1', errors='ignore')

    interval = max(0.0, args.interval or 0.0)
    remaining = args.count if args.count is not None else 1
    sample_idx = 0

    while True:
        sample_idx += 1
        stats = _AlglibCoreStats()
        status = gpu_get_stats(ctypes.byref(stats))
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        if status != 0:
            print(f"[{timestamp}] gpu_get_stats failed: {status_to_string(status)} ({status})")
            return status

        rows = []
        if version_label:
            rows.append(('Version', version_label))
        rows.extend([
            ('Backend', _backend_name(stats.backend_type)),
            ('Workers', str(stats.worker_threads)),
            ('Queue', f"{stats.queue_depth}/{stats.queue_limit}"),
            ('Inflight', str(stats.inflight_jobs)),
            ('Submitted', str(stats.total_jobs_submitted)),
            ('Completed', str(stats.total_jobs_completed)),
            ('Avg Duration (ms)', str(getattr(stats, 'avg_job_duration_ms', 0))),
            ('Last Duration (ms)', str(getattr(stats, 'last_job_duration_ms', 0))),
            ('Last Status', f"{status_to_string(getattr(stats, 'last_status', 0))} ({getattr(stats, 'last_status', 0)})"),
            ('Last Update', _format_last_update(stats)),
            ('Sampled', f"{timestamp} UTC"),
        ])

        print_table(rows, ('Field', 'Value'))

        if interval <= 0.0:
            break
        if remaining is not None and remaining > 0:
            if sample_idx >= remaining:
                break
        time.sleep(interval)

    return 0


def kill_processes(names):
    if not is_windows():
        return
    for n in names:
        # taskkill ignores if not found
        try:
            run(['taskkill', '/F', '/IM', n], check=False)
        except Exception:
            pass

def ps_list_procs_with_modules(mod_names):
    """Return list of dicts with processes that have any of mod_names loaded as module filename."""
    if not is_windows():
        return []
    try:
        mods = ';'.join(mod_names)
        script = (
            "$mods = '" + mods + "'.Split(';');"
            "$res = @();"
            "Get-Process | ForEach-Object { $p = $_; try { $_.Modules | ForEach-Object { $m=$_; $base=[System.IO.Path]::GetFileName($m.FileName); if($mods -contains $base){ $res += [PSCustomObject]@{ Id=$p.Id; Name=$p.ProcessName; Module=$base } } } } catch {} } ;"
            "$res | ConvertTo-Json -Compress"
        )
        r = run(['powershell.exe','-NoProfile','-Command', script], capture_output=True, text=True, check=False)
        out = (r.stdout or '').strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            return [data]
        return data
    except Exception:
        return []


def cmd_kill(_args):
    # Funciona em Windows e WSL (via powershell/taskkill)
    names = ['terminal64.exe','terminal.exe','metaeditor64.exe','metaeditor.exe']
    try:
        if is_windows():
            kill_processes(names)
            return 0
        if is_wsl():
            # Tentar taskkill via powershell.exe
            for n in names:
                try:
                    subprocess.run(['powershell.exe','-NoProfile','-Command', f"taskkill /F /IM {n} /T"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            print('kill: solicitado via taskkill (WSL)')
            return 0
    except Exception as e:
        print('kill: erro', e)
    return 1
    return 0


def print_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = '  '.join('{:' + str(w) + '}' for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*['-' * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


RELEASE_REQUIRED_FILES = [
    ('mt-bridge.dll', 'Bridge DLL'),
    ('core.dll', 'Core DLL'),
    ('tester.dll', 'Tester alias'),
]


def report_release_artifacts(release_dir: str):
    release_abs = os.path.abspath(release_dir)
    if not os.path.isdir(release_abs):
        disp = to_winpath(release_abs) if is_wsl() else release_abs
        print(f"\nRelease artifacts: directory not found -> {disp}")
        return
    rows = []
    for fname, label in RELEASE_REQUIRED_FILES:
        path = os.path.join(release_abs, fname)
        disp = to_winpath(path) if is_wsl() else path
        if os.path.isfile(path):
            st = os.stat(path)
            rows.append([label, fname, 'OK', fmt_dt(st.st_mtime), disp])
        else:
            rows.append([label, fname, 'MISSING', '-', disp])
    print('\nRelease artifacts:')
    print_table(rows, headers=['Artifact','File','State','Modified','Path'])


def cmd_link(args):
    if not (is_windows() or is_wsl()):
        print('ERROR: link command only supported on Windows/WSL.')
        return 2

    root = cli_root_default()
    release = _normalize_path(args.release) if args.release else os.path.join(root, 'build-win', args.config or 'Release')
    release = os.path.abspath(release)
    if not os.path.isdir(release):
        print('ERROR: release directory not found:', release)
        return 3

    manual_libs = getattr(args, 'libs', None)
    targets = []
    if manual_libs:
        targets = [{'id': f'manual{idx+1}', 'libs': _normalize_path(p)} for idx, p in enumerate(manual_libs)]
    else:
        pid, proj_libs, err = resolve_project_libs(root, getattr(args, 'project', None))
        if not proj_libs:
            print('ERROR:', err)
            return 6
        targets = [{'id': pid, 'libs': proj_libs}]

    if getattr(args, 'agents', False):
        agent_list = find_agent_libs()
        if agent_list:
            base = len(targets)
            for idx, p in enumerate(agent_list):
                targets.append({'id': f'agent{base+idx+1}', 'libs': p})

    if not targets:
        print('No target Libraries resolved.')
        return 7

    release_disp = to_winpath(release) if is_wsl() else release
    print(f"\n[link] release dir: {release_disp}")
    rows = []
    errors = 0
    for entry in targets:
        libs = _normalize_path(entry['libs'])
        state, note = _ensure_link(libs, release)
        if state == 'ERROR':
            errors += 1
        rows.append([entry.get('id', '?'), to_winpath(libs) if is_wsl() else libs, state, note])

    print('\nLink state:')
    print_table(rows, headers=['TerminalID','Libraries','State','Note'])
    report_release_artifacts(release)
    return 0 if errors == 0 else 8


def main():
    ap = argparse.ArgumentParser(prog='mtcli', description='MetaTrader CLI for ALGLIB bridge')
    sub = ap.add_subparsers(dest='cmd', required=False)

    p = sub.add_parser('env', help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser('detect', help=argparse.SUPPRESS)
    p.add_argument('--project', help='Project id to inspect (default: last saved project)')
    p.add_argument('--all', action='store_true', help='List every detected Terminal/Agent instead of just the project libs')
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser('list', help=argparse.SUPPRESS)
    p.add_argument('--include-release', action='store_true', help='Also check dist-wave/Release (prints explicit FALLBACK when used)')
    p.set_defaults(func=cmd_list_v2)

    p = sub.add_parser('ping', help=argparse.SUPPRESS)
    p.add_argument('--dll', help='Path to mt-bridge.dll (auto-detect if omitted)')
    p.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
    p.add_argument('--streams', type=int, default=4, help='Stream count (default: 4)')
    p.set_defaults(func=cmd_ping)

    p = sub.add_parser('config', help='Show or update mtcli configuration')
    cfg_sub = p.add_subparsers(dest='ccmd', required=False)
    def _cfg_lang_set(args):
        root = cli_root_default(); cfg = load_config(root)
        value = (args.to or 'en').lower()
        if value not in ('en','pt','pt-br','pt_pt'):
            print('Invalid language. Use: en or pt'); return 2
        cfg['lang'] = 'pt' if value.startswith('pt') else 'en'
        save_config(root, cfg)
        print('Language set to:', cfg['lang'])
        return 0
    pcfg_lang = cfg_sub.add_parser('lang', help='Language settings (en/pt)')
    pcfg_lang_sub = pcfg_lang.add_subparsers(dest='lcmd', required=True)
    pls = pcfg_lang_sub.add_parser('set', help='Set language (en/pt)'); pls.add_argument('--to', required=True); pls.set_defaults(func=_cfg_lang_set)
    def _cfg_lang_show(_args):
        print('Language:', get_lang()); return 0
    plsh = pcfg_lang_sub.add_parser('show', help='Show current language'); plsh.set_defaults(func=_cfg_lang_show)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser('fft', help=argparse.SUPPRESS)
    p.add_argument('--dll', help='Path to mt-bridge.dll (auto-detect if omitted)')
    p.add_argument('--size', type=int, default=1024, help='FFT size (default: 1024)')
    p.add_argument('--repeat', type=int, default=1, help='How many times to run the FFT (default: 1)')
    p.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
    p.add_argument('--streams', type=int, default=4, help='Stream count (default: 4)')
    p.set_defaults(func=cmd_fft)

    p = sub.add_parser('badheader', help='Check indicator header for required directives/imports')
    p.add_argument('--indicator', help='Path to indicator (default: Indicators/Conexao_ALGLIB.mq5)')
    p.set_defaults(func=cmd_badheader)

    p = sub.add_parser('meta', help=argparse.SUPPRESS)
    p.add_argument('--indicator', required=True, help='Path to .mq5 file to compile')
    p.add_argument('--metaeditor', help='Path to metaeditor64.exe')
    p.add_argument('--log', help='Path to log file (default: same folder)')
    p.set_defaults(func=cmd_meta)

    def add_mt_args(parser):
        parser.add_argument('--project', help='Projeto salvo (default: last)')
        parser.add_argument('--libs', help='Caminho manual para MQL5\\Libraries')
        parser.add_argument('--terminal', help='Caminho manual para terminal64.exe')
        parser.add_argument('--metaeditor', help='Caminho manual para metaeditor64.exe')
        parser.add_argument('--data-dir', help='Data Folder raiz (pasta contendo MQL5)')
        parser.add_argument('--portable', type=int, choices=[0,1], help='Modo Portable (1) ou /datapath (0); default: automático por projeto')
        parser.add_argument('--profile', help='Profile do Terminal a usar (default: do projeto ou Default)')

    # install (indicator/expert)
    pinstall = sub.add_parser('install', help=argparse.SUPPRESS)
    install_sub = pinstall.add_subparsers(dest='install_cmd', required=True)
    pi = install_sub.add_parser('indicator', help='Instala um indicador (.mq5/.ex5)')
    pi.add_argument('--file', required=True, help='Arquivo de origem (.mq5/.ex5)')
    pi.add_argument('--no-compile', action='store_true', help='Não compilar após copiar (.mq5)')
    add_mt_args(pi)
    pi.set_defaults(func=cmd_install_indicator)
    pe = install_sub.add_parser('expert', help='Instala um expert (.mq5/.ex5)')
    pe.add_argument('--file', required=True, help='Arquivo de origem (.mq5/.ex5)')
    pe.add_argument('--no-compile', action='store_true', help='Não compilar após copiar (.mq5)')
    add_mt_args(pe)
    pe.set_defaults(func=cmd_install_expert)

    # listener
    plistener = sub.add_parser('listener', help='Gerencia CommandListenerEA e comandos')
    listener_sub = plistener.add_subparsers(dest='listener_cmd', required=True)
    li = listener_sub.add_parser('install', help='Instala/compila CommandListenerEA + scripts')
    li.add_argument('--force', action='store_true', help='Sobrescreve arquivos existentes')
    add_mt_args(li)
    li.set_defaults(func=cmd_listener_install)
    lr = listener_sub.add_parser('run', help='Inicia o terminal com CommandListenerEA')
    lr.add_argument('--symbol', default='EURUSD')
    lr.add_argument('--period', default='H1')
    lr.add_argument('--indicator', help='Anexar indicador automaticamente ao iniciar')
    lr.add_argument('--indicator-subwindow', type=int, default=0)
    lr.add_argument('--ini', help='Arquivo .ini para salvar (default: ./listener.ini)')
    lr.add_argument('--trace', action='store_true', help='Mostra a linha de comando usada para iniciar o Terminal')
    g_det = lr.add_mutually_exclusive_group()
    g_det.add_argument('--detach', dest='detach', action='store_true', default=True, help='Inicia e retorna imediatamente (default)')
    g_det.add_argument('--wait', dest='detach', action='store_false', help='Aguarda o Terminal encerrar')
    add_mt_args(lr)
    lr.set_defaults(func=cmd_listener_run)

    ls = listener_sub.add_parser('status', help='Envia PING e mostra logs para conferir se o listener está ativo')
    add_mt_args(ls)
    ls.set_defaults(func=cmd_listener_status)

    le = listener_sub.add_parser('ensure', help='Instala (se preciso) e inicia o listener de forma não-bloqueante')
    le.add_argument('--force-start', action='store_true', help='Sempre iniciar nova instância do Terminal (ignora PING)')
    add_mt_args(le)
    le.set_defaults(func=cmd_listener_ensure)



    # chart commands
    pchart = sub.add_parser('chart', help=argparse.SUPPRESS)
    chart_sub = pchart.add_subparsers(dest='chart_domain', required=True)

    pci = chart_sub.add_parser('indicator', help='Opera indicadores no gráfico')
    ci_sub = pci.add_subparsers(dest='chart_action', required=True)
    cia = ci_sub.add_parser('attach', help='Anexa indicador via CommandListenerEA')
    cia.add_argument('--symbol', default='EURUSD')
    cia.add_argument('--period', default='H1')
    cia.add_argument('--indicator', required=True, help='Nome do indicador (sem .mq5)')
    cia.add_argument('--subwindow', type=int, default=0)
    add_mt_args(cia)
    cia.set_defaults(func=chart_indicator_attach)
    cidet = ci_sub.add_parser('detach', help='Remove indicador do gráfico')
    cidet.add_argument('--symbol', default='EURUSD')
    cidet.add_argument('--period', default='H1')
    cidet.add_argument('--indicator', required=True)
    cidet.add_argument('--subwindow', type=int, default=0)
    add_mt_args(cidet)
    cidet.set_defaults(func=chart_indicator_detach)

    # indicator ensure (instala/compila se necessário e opcionalmente anexa)
    pind = sub.add_parser('indicator', help=argparse.SUPPRESS)
    ind_sub = pind.add_subparsers(dest='icmd', required=True)
    ien = ind_sub.add_parser('ensure', help='Garante indicador instalado/compilado e pode anexar em um gráfico')
    ien.add_argument('--file', required=True, help='Caminho do .mq5/.ex5')
    ien.add_argument('--attach', action='store_true', help='Anexar ao gráfico após instalar')
    ien.add_argument('--symbol', default='EURUSD')
    ien.add_argument('--period', default='H1')
    ien.add_argument('--subwindow', type=int, default=1)
    add_mt_args(ien)
    ien.set_defaults(func=cmd_indicator_ensure)

    pce = chart_sub.add_parser('expert', help='Opera experts no gráfico')
    ce_sub = pce.add_subparsers(dest='chart_exp_action', required=True)
    cea = ce_sub.add_parser('attach', help='Anexa Expert Advisor')
    cea.add_argument('--symbol', default='EURUSD')
    cea.add_argument('--period', default='H1')
    cea.add_argument('--expert', required=True, help='Caminho relativo (ex.: Pasta\\MeuEA)')
    cea.add_argument('--template', help='Nome do template já existente em Templates')
    cea.add_argument('--template-src', help='Arquivo .tpl para copiar e usar')
    add_mt_args(cea)
    cea.set_defaults(func=chart_expert_attach)
    ced = ce_sub.add_parser('detach', help='Remove Expert Advisor')
    ced.add_argument('--symbol', default='EURUSD')
    ced.add_argument('--period', default='H1')
    add_mt_args(ced)
    ced.set_defaults(func=chart_expert_detach)

    # screenshot do gráfico
    pcs = chart_sub.add_parser('screenshot', help='Captura um screenshot do gráfico (dual-mode)')
    pcs.add_argument('--symbol', default='EURUSD')
    pcs.add_argument('--period', default='H1')
    pcs.add_argument('--width', type=int, default=0, help='Largura (0 = auto)')
    pcs.add_argument('--height', type=int, default=0, help='Altura (0 = auto)')
    pcs.add_argument('--output', help='Arquivo de saída (.png). Relativo → MQL5/Files/screenshots')
    add_mt_args(pcs)
    pcs.set_defaults(func=chart_screenshot)

    # ---- dll (núcleo em DLLs e inspeção) ----
    pdll = sub.add_parser('dll', help='Operações sobre DLLs (build/link/inspeção GPU/bridge)')
    sdll = pdll.add_subparsers(dest='dcmd', required=True)

    d_build = sdll.add_parser('build', help='Configure+build via CMake; optional preset regen')
    d_build.add_argument('--build-dir', help='Build directory (default: <root>/build-win)')
    d_build.add_argument('--arch', default='x64', help='Windows arch for CMake -A (default: x64)')
    d_build.add_argument('--config', default='Release', help='Build config (default: Release)')
    d_build.add_argument('--targets', nargs='+', help='Build targets (default: ALL)')
    d_build.add_argument('--parallel', type=int, default=os.cpu_count() or 8, help='-j (default: cpu count)')
    d_build.add_argument('--preset', help='Run opgen.py to (re)generate preset include and exports')
    d_build.add_argument('--use-preset-selection', action='store_true', help='Use exactly preset ops without prompt')
    d_build.add_argument('--no-union-presets', action='store_true', help='Do not union presets for exports')
    d_build.add_argument('--generator', help='CMake generator name (default: Visual Studio 17 2022 on Windows)')
    d_build.add_argument('--toolset', help='CMake toolset string (default: host=x64 when using vcvarsall)')
    d_build.add_argument('--no-vcvars', action='store_true', help='Skip vcvarsall.bat even on Windows')
    d_build.add_argument('--vcvars', help='Path to vcvarsall.bat (default: VS2022 BuildTools)')
    d_build.add_argument('extra', nargs=argparse.REMAINDER, help='Extra args after -- passed to cmake --build')
    d_build.set_defaults(func=cmd_build)

    d_ba = sdll.add_parser('build-all', help='Generate (glue+exports) then build Core, Bridge (Release)')
    d_ba.add_argument('--arch', default='x64')
    d_ba.add_argument('--config', default='Release')
    d_ba.add_argument('--parallel', type=int, default=os.cpu_count() or 8)
    d_ba.add_argument('--preset', help='Optional preset name to generate indicator stub')
    d_ba.set_defaults(func=cmd_build_all)

    d_link = sdll.add_parser('link', help='Create/validate junctions MQL5\\Libraries → build-win\\<config>')
    d_link.add_argument('--release', help='Explicit Release directory (default: build-win/<config>)')
    d_link.add_argument('--config', default='Release', help='Build configuration folder')
    d_link.add_argument('--libs', action='append', help='Explicit MQL5\\Libraries (repeatable)')
    d_link.add_argument('--project', help='Project id to use when no --libs is provided (default: last)')
    d_link.add_argument('--agents', action='store_true', help='Also link Tester Agents')
    d_link.set_defaults(func=cmd_link)

    d_pkg = sdll.add_parser('package', help='Empacota binários/resultados')
    d_pkg.add_argument('--version', required=True, help='Release version tag (e.g., 0.2.0)')
    d_pkg.add_argument('--output', help='Output directory (default: dist/releases)')
    d_pkg.set_defaults(func=cmd_package)

    d_fetch = sdll.add_parser('fetch-cuda', help='Baixa redistribuíveis CUDA')
    d_fetch.set_defaults(func=cmd_fetch_cuda)

    d_ping = sdll.add_parser('ping', help='Versão/estatísticas da bridge (DLL)')
    d_ping.add_argument('--dll', help='Path to mt-bridge.dll (auto-detect if omitted)')
    d_ping.add_argument('--device', type=int, default=0)
    d_ping.add_argument('--streams', type=int, default=4)
    d_ping.set_defaults(func=cmd_ping)

    d_fft = sdll.add_parser('fft', help='FFT de sanidade via DLL')
    d_fft.add_argument('--dll', help='Path to mt-bridge.dll (auto-detect if omitted)')
    d_fft.add_argument('--size', type=int, default=1024)
    d_fft.add_argument('--repeat', type=int, default=1)
    d_fft.add_argument('--device', type=int, default=0)
    d_fft.add_argument('--streams', type=int, default=4)
    d_fft.set_defaults(func=cmd_fft)

    d_hb = sdll.add_parser('heartbeat', help='Sondagem periódica de stats da GPU (DLL)')
    d_hb.add_argument('--dll', help='Path to mt-bridge DLL/SO (auto-detect if omitted)')
    d_hb.add_argument('--interval', type=float, default=0.0)
    d_hb.add_argument('--count', type=int, default=1)
    d_hb.set_defaults(func=cmd_heartbeat)

    # dll list (alias for top-level list)
    d_list = sdll.add_parser('list', help='List build artifacts (Release)')
    d_list.add_argument('--include-release', action='store_true', help='Also check dist-wave/Release (prints explicit FALLBACK when used)')
    d_list.set_defaults(func=cmd_list_v2)

    # dll gen-mql (alias for top-level gen-mql)
    d_gen = sdll.add_parser('gen-mql', help='Generate MQL5 indicator/script and optionally copy to Terminals')
    d_gen.add_argument('--type', choices=['indicator','script'], default='indicator', help='MQL output type')
    d_gen.add_argument('--name', help='Output name (without extension). Default: Conexao_ALGLIB')
    d_gen.add_argument('--yaml', help='YAML path with selection (ops: [...])')
    d_gen.add_argument('--preset', help='Optional preset label (metadata only)')
    d_gen.add_argument('--copy', action='store_true', help='Copy generated file to Terminal(s) MQL5 folder')
    d_gen.add_argument('--terminal-id', help='Specific Terminal ID to copy to (default: all)')
    d_gen.add_argument('--all', action='store_true', help='Copy to all Terminals')
    d_gen.add_argument('--no-from-exports', action='store_true', help='Do not scan dll_exports.cpp; use YAML/preset only')
    d_gen.add_argument('--mql5-root', help='Explicit MQL5 root (e.g. C:/mql5/MQL5) when copying generated MQL')
    d_gen.set_defaults(func=cmd_gen_mql)

    # ---- build (top-level legacy; mantém por compatibilidade) ----
    p = sub.add_parser('build', help=argparse.SUPPRESS)
    p.add_argument('--build-dir', help='Build directory (default: <root>/build-win)')
    p.add_argument('--arch', default='x64', help='Windows arch for CMake -A (default: x64)')
    p.add_argument('--config', default='Release', help='Build config (default: Release)')
    p.add_argument('--targets', nargs='+', help='Build targets (default: ALL)')
    p.add_argument('--parallel', type=int, default=os.cpu_count() or 8, help='-j (default: cpu count)')
    p.add_argument('--preset', help='Run opgen.py to (re)generate preset include and exports')
    p.add_argument('--use-preset-selection', action='store_true', help='Use exactly preset ops without prompt')
    p.add_argument('--no-union-presets', action='store_true', help='Do not union presets for exports')
    p.add_argument('--generator', help='CMake generator name (default: Visual Studio 17 2022 on Windows)')
    p.add_argument('--toolset', help='CMake toolset string (default: host=x64 when using vcvarsall)')
    p.add_argument('--no-vcvars', action='store_true', help='Skip vcvarsall.bat even on Windows')
    p.add_argument('--vcvars', help='Path to vcvarsall.bat (default: VS2022 BuildTools)')
    p.add_argument('extra', nargs=argparse.REMAINDER, help='Extra args after -- passed to cmake --build')
    p.set_defaults(func=cmd_build)

    pba = sub.add_parser('build-all', help='[LEGACY] Use dll build-all')
    pba.add_argument('--arch', default='x64')
    pba.add_argument('--config', default='Release')
    pba.add_argument('--parallel', type=int, default=os.cpu_count() or 8)
    pba.add_argument('--preset', help='Optional preset name to generate indicator stub')
    pba.set_defaults(func=cmd_build_all)

    phb = sub.add_parser('heartbeat', help='[LEGACY] Use dll heartbeat')
    phb.add_argument('--dll', help='Path to mt-bridge DLL/SO (auto-detect if omitted)')
    phb.add_argument('--interval', type=float, default=0.0, help='Seconds between samples (0 = single read)')
    phb.add_argument('--count', type=int, default=1, help='Samples to capture when interval > 0 (0 = infinite)')
    phb.set_defaults(func=cmd_heartbeat)

    ptst = sub.add_parser('tester', help='Run automated tester harness')
    ptst.add_argument('--config', help='Path to tester config JSON')
    ptst.set_defaults(func=cmd_tester)

    pmatrix = sub.add_parser('matrix', help=argparse.SUPPRESS)
    pmatrix.add_argument('--workflow', help='Workflow path (default: .github/workflows/matrix-ci.yml)')
    pmatrix.set_defaults(func=cmd_matrix)

    ppack = sub.add_parser('package', help=argparse.SUPPRESS)
    ppack.add_argument('--version', required=True, help='Release version tag (e.g., 0.2.0)')
    ppack.add_argument('--output', help='Output directory (default: dist/releases)')
    ppack.set_defaults(func=cmd_package)

    p = sub.add_parser('kill', help='Kill Terminal/MetaEditor/Service processes')
    p.set_defaults(func=cmd_kill)

    # ---- code (desenvolvimento) ----
    pcode = sub.add_parser('code', help=argparse.SUPPRESS)
    scode = pcode.add_subparsers(dest='ccmd', required=True)
    cc = scode.add_parser('compile', help='Compila um arquivo .mq5 via MetaEditor (tail do log)')
    cc.add_argument('--file', required=True, help='Caminho do arquivo .mq5')
    cc.add_argument('--include', action='append', help='Pasta extra de include (repetível)')
    cc.add_argument('--syntax-only', action='store_true', help='Apenas verificação de sintaxe (equivale a /s)')
    add_mt_args(cc)
    def cmd_code_compile(args):
        # Baseia-se no metaeditor salvo no projeto
        _, metaeditor, _ = resolve_mt_context(args)
        src = Path(args.file)
        if not src.exists():
            raise SystemExit(f"Fonte não encontrado: {src}")
        log_path = src.with_suffix('.log')
        # Monta argumentos /compile /log [/s] [/include:]
        argv = [f"/compile:{to_windows_path(src)}", f"/log:{to_windows_path(log_path)}"]
        if getattr(args, 'syntax_only', False):
            argv.append('/s')
        for inc in (getattr(args, 'include', None) or []):
            argv.append(f"/include:{to_windows_path(inc)}")
        rc = run_win_exe(metaeditor, argv, detach=False)
        print_compile_log_tail(log_path, tag='MetaEditor (code compile)')
        return rc
    cc.set_defaults(func=cmd_code_compile)

    # ---- Principais (sempre simples) ----
    pr = sub.add_parser('run', help=argparse.SUPPRESS)
    pr.add_argument('--symbol', help='Símbolo (default: do projeto)')
    pr.add_argument('--period', help='Timeframe (default: do projeto)')
    add_mt_args(pr)
    def cmd_top_run(args):
        with_project_defaults(args, use_indicator=False)
        le = argparse.Namespace(project=getattr(args,'project',None), libs=None, terminal=None, metaeditor=None, data_dir=None, force=False, force_start=False,
                                symbol=args.symbol, period=args.period)
        return cmd_listener_ensure(le)
    pr.set_defaults(func=cmd_top_run)

    pa = sub.add_parser('attach', help=argparse.SUPPRESS)
    pa.add_argument('--indicator', help='Nome do indicador (default: do projeto)')
    pa.add_argument('--file', help='Arquivo .mq5/.ex5 (opcional, para instalar/compilar antes de anexar)')
    pa.add_argument('--symbol', help='Símbolo (default: do projeto)')
    pa.add_argument('--period', help='Timeframe (default: do projeto)')
    pa.add_argument('--subwindow', type=int, help='Subjanela (default: do projeto)')
    add_mt_args(pa)
    def cmd_top_attach(args):
        # Se foi fornecido --file, instala/compila e em seguida anexa; caso contrário, apenas anexa
        with_project_defaults(args, use_indicator=True)
        if getattr(args, 'file', None):
            ien = argparse.Namespace(project=getattr(args,'project',None), libs=None, terminal=None, metaeditor=None, data_dir=None,
                                     file=args.file, attach=True, symbol=args.symbol, period=args.period, subwindow=args.subwindow)
            return cmd_indicator_ensure(ien)
        a = argparse.Namespace(project=getattr(args,'project',None), libs=None, terminal=None, metaeditor=None, data_dir=None,
                               symbol=args.symbol, period=args.period, indicator=args.indicator, subwindow=args.subwindow)
        return chart_indicator_attach(a)
    pa.set_defaults(func=cmd_top_attach)

    pd = sub.add_parser('detach', help=argparse.SUPPRESS)
    pd.add_argument('--indicator')
    pd.add_argument('--symbol')
    pd.add_argument('--period')
    pd.add_argument('--subwindow', type=int)
    add_mt_args(pd)
    def cmd_top_detach(args):
        a = argparse.Namespace(project=getattr(args,'project',None), libs=None, terminal=None, metaeditor=None, data_dir=None,
                               symbol=args.symbol, period=args.period, indicator=args.indicator, subwindow=args.subwindow)
        return chart_indicator_detach(a)
    pd.set_defaults(func=cmd_top_detach)

    ps = sub.add_parser('status', help=argparse.SUPPRESS)
    add_mt_args(ps)
    def cmd_top_status(args):
        return cmd_listener_status(args)
    ps.set_defaults(func=cmd_top_status)

    # Alias simples para screenshot no mesmo nível
    pshot = sub.add_parser('screenshot', help=argparse.SUPPRESS)
    pshot.add_argument('--symbol', help='Símbolo (default: do projeto)')
    pshot.add_argument('--period', help='Timeframe (default: do projeto)')
    pshot.add_argument('--width', type=int, help='Largura (0 = auto)')
    pshot.add_argument('--height', type=int, help='Altura (0 = auto)')
    pshot.add_argument('--output', help='Arquivo de saída (.png)')
    add_mt_args(pshot)
    def cmd_top_screenshot(args):
        with_project_defaults(args, use_indicator=False)
        a = argparse.Namespace(project=getattr(args,'project',None), libs=None, terminal=None, metaeditor=None, data_dir=None,
                               symbol=args.symbol, period=args.period, width=getattr(args,'width',0) or 0,
                               height=getattr(args,'height',0) or 0, output=getattr(args,'output',None))
        return chart_screenshot(a)
    pshot.set_defaults(func=cmd_top_screenshot)

    # ---- mt (grupo de controle do MT5) ----
    # Mantido como alias por compatibilidade; o preferido é 'terminal'.
    pmt = sub.add_parser('mt', help='[ALIAS] use o grupo terminal (status/chart/tester/logs)')
    smt = pmt.add_subparsers(dest='mtcmd', required=True)

    mtst = smt.add_parser('status', help='Mostra saúde do Terminal/Listener e tail de logs')
    mtst.add_argument('--repair', action='store_true', help='Corrige ausências (inicia Terminal/instala listener)')
    add_mt_args(mtst)
    def cmd_mt_status(args):
        if getattr(args, 'repair', False):
            # Usa ensure para (re)instalar e (re)iniciar, mas aborta se a compilação do listener falhar
            e = argparse.Namespace(project=getattr(args,'project',None), root=getattr(args,'root',None), libs=None, terminal=None, metaeditor=None, data_dir=None, force=True, force_start=False,
                                   portable=getattr(args,'portable',None), profile=getattr(args,'profile',None))
            rc = cmd_listener_ensure(e)
            if rc != 0:
                return rc
        return cmd_listener_status(args)
    mtst.set_defaults(func=cmd_mt_status)

    # mt verify-datapath — verifica se o MT iniciou exatamente na Data Folder do projeto
    def cmd_mt_verify_datapath(args):
        with_project_defaults(args, use_indicator=False)
        terminal, metaeditor, data_dir = resolve_mt_context(args)
        ensure_source(metaeditor, data_dir, 'Scripts/DumpDataPath.mq5', SCRIPT_DUMP_DATAPATH, force=True, quiet=False)
        ini = Path('verify_datapath.ini').absolute()
        content = build_ini_startup(symbol=args.symbol, period=timeframe_ok(args.period), template=None,
                                    expert=None, script='DumpDataPath', expert_params=None, script_params=None,
                                    shutdown=False)
        write_text_utf16(ini, content)
        extra = [f"/config:{to_windows_path(ini)}"]
        prof = getattr(args, 'profile', None) or 'Default'
        extra.append(f"/profile:{prof}")
        use_portable, dp_root = _should_use_portable(terminal, data_dir, getattr(args, 'portable', None))
        if use_portable:
            extra.append('/portable')
        else:
            extra.append(f"/datapath:{to_windows_path(dp_root)}")
        run_win_exe(terminal, extra, detach=True)
        # Espera robusta pelo arquivo (até ~5s)
        out_file = win_to_wsl(data_dir) / 'Files' / 'datapath.txt'
        reported = ''
        for _ in range(20):
            if out_file.exists():
                try:
                    reported = out_file.read_text(encoding='ascii', errors='ignore').strip()
                    if reported:
                        break
                except Exception:
                    pass
            time.sleep(0.25)
        mql5 = _mql5_dir_from_data_dir(data_dir)
        expected_root = str(mql5.parent)
        print(f"[verify] reported: {reported}")
        print(f"[verify] expected root: {expected_root}")
        ok = reported.replace('\\','/').lower().strip() == expected_root.replace('\\','/').lower().strip()
        print("[verify] RESULT:", "OK" if ok else "MISMATCH")
        print_log_tail('verify datapath', data_dir=data_dir)
        print(tr('done'))
        return 0 if ok else 2

    mv = smt.add_parser('verify-datapath', help='Verifica se o MT iniciou exatamente na Data Folder do projeto')
    mv.add_argument('--symbol', default='EURUSD')
    mv.add_argument('--period', default='H1')
    add_mt_args(mv)
    mv.set_defaults(func=cmd_mt_verify_datapath)

    # ---- terminal (alias estruturado de 'mt') ----
    pterm = sub.add_parser('terminal', help='Terminal/MT5 control (status/chart/create/screenshot/tester)')
    ster = pterm.add_subparsers(dest='tcmd', required=True)

    # terminal status (alias)
    t_status = ster.add_parser('status', help='Show Terminal/Listener health and tail logs')
    add_mt_args(t_status)
    t_status.set_defaults(func=cmd_mt_status)

    # terminal verify-datapath (alias)
    t_vd = ster.add_parser('verify-datapath', help='Verify that MT started in the exact project Data Folder')
    t_vd.add_argument('--symbol', default='EURUSD')
    t_vd.add_argument('--period', default='H1')
    add_mt_args(t_vd)
    t_vd.set_defaults(func=cmd_mt_verify_datapath)

    # terminal screenshot (alias chart screenshot)
    t_shot = ster.add_parser('screenshot', help='Capture chart screenshot (dual-mode)')
    t_shot.add_argument('--symbol')
    t_shot.add_argument('--period')
    t_shot.add_argument('--width', type=int)
    t_shot.add_argument('--height', type=int)
    t_shot.add_argument('--output')
    add_mt_args(t_shot)
    def _term_screenshot(args):
        with_project_defaults(args, use_indicator=False)
        a = argparse.Namespace(project=getattr(args,'project',None), libs=None, terminal=None, metaeditor=None, data_dir=None,
                               symbol=args.symbol, period=args.period, width=getattr(args,'width',0) or 0,
                               height=getattr(args,'height',0) or 0, output=getattr(args,'output',None))
        return chart_screenshot(a)
    t_shot.set_defaults(func=_term_screenshot)

    # terminal chart (reuse mt chart tree)
    t_chart = ster.add_parser('chart', help='Chart operations (indicator|expert|template)')
    tch = t_chart.add_subparsers(dest='chart', required=True)
    # indicator
    tci = tch.add_parser('indicator', help='Attach/detach indicators')
    tci_sub = tci.add_subparsers(dest='act', required=True)
    _tia = tci_sub.add_parser('attach', help='Attach indicator')
    _tia.add_argument('--indicator', required=True)
    _tia.add_argument('--symbol'); _tia.add_argument('--period'); _tia.add_argument('--subwindow', type=int)
    add_mt_args(_tia); _tia.set_defaults(func=chart_indicator_attach)
    _tid = tci_sub.add_parser('detach', help='Detach indicator')
    _tid.add_argument('--indicator', required=True)
    _tid.add_argument('--symbol'); _tid.add_argument('--period'); _tid.add_argument('--subwindow', type=int)
    add_mt_args(_tid); _tid.set_defaults(func=chart_indicator_detach)
    # expert
    tce = tch.add_parser('expert', help='Attach/detach Expert Advisor (via template)')
    tce_sub = tce.add_subparsers(dest='act', required=True)
    _tea = tce_sub.add_parser('attach', help='Attach EA (uses template)')
    _tea.add_argument('--expert', required=True); _tea.add_argument('--template'); _tea.add_argument('--template-src')
    _tea.add_argument('--symbol'); _tea.add_argument('--period'); add_mt_args(_tea); _tea.set_defaults(func=chart_expert_attach)
    _ted = tce_sub.add_parser('detach', help='Detach EA')
    _ted.add_argument('--symbol'); _ted.add_argument('--period'); add_mt_args(_ted); _ted.set_defaults(func=chart_expert_detach)
    # template
    tct = tch.add_parser('template', help='Apply template')
    tct.add_argument('--template', required=True); tct.add_argument('--symbol'); tct.add_argument('--period'); add_mt_args(tct)
    tct.set_defaults(func=chart_template_apply)

    # terminal create (replacement of install)
    def _term_create_indicator(args):
        return cmd_install_indicator(args)
    def _term_create_expert(args):
        return cmd_install_expert(args)
    def _term_create_script(args):
        # reuse mt install script
        return cmd_mt_install_script(args)
    def _term_create_template(args):
        return cmd_mt_install_template(args)
    def _term_create_ini(args):
        return cmd_mt_ini_create(args)

    tcreate = ster.add_parser('create', help='Create/copy artifacts into the project Data Folder')
    tc = tcreate.add_subparsers(dest='what', required=True)
    tci2 = tc.add_parser('indicator', help='Copy/compile indicator into MQL5/Indicators')
    tci2.add_argument('--file', required=True); tci2.add_argument('--no-compile', action='store_true'); add_mt_args(tci2)
    tci2.set_defaults(func=_term_create_indicator)
    tce2 = tc.add_parser('expert', help='Copy/compile expert into MQL5/Experts')
    tce2.add_argument('--file', required=True); tce2.add_argument('--no-compile', action='store_true'); add_mt_args(tce2)
    tce2.set_defaults(func=_term_create_expert)
    tcs2 = tc.add_parser('script', help='Copy/compile script into MQL5/Scripts')
    tcs2.add_argument('--file', required=True); tcs2.add_argument('--no-compile', action='store_true'); add_mt_args(tcs2)
    tcs2.set_defaults(func=_term_create_script)
    tct2 = tc.add_parser('template', help='Copy template into Profiles/Templates')
    tct2.add_argument('--file', required=True); add_mt_args(tct2); tct2.set_defaults(func=_term_create_template)
    tci_ini = tc.add_parser('ini', help='Generate .ini (StartUp) for boot')
    tci_ini.add_argument('--symbol'); tci_ini.add_argument('--period'); tci_ini.add_argument('--expert', default='CommandListenerEA')
    tci_ini.add_argument('--template'); tci_ini.add_argument('--script'); tci_ini.add_argument('--output', default='listener.ini'); add_mt_args(tci_ini)
    tci_ini.set_defaults(func=_term_create_ini)

    # terminal kill (alias)
    tkill = ster.add_parser('kill', help='Kill Terminal/MetaEditor processes')
    tkill.set_defaults(func=cmd_kill)

    # ---- metaeditor (rename of 'code') ----
    pme = sub.add_parser('metaeditor', help='MetaEditor operations (compile etc.)')
    me_sub = pme.add_subparsers(dest='mecmd', required=True)
    me_c = me_sub.add_parser('compile', help='Compile indicator/EA via MetaEditor')
    me_c.add_argument('--file', required=True, help='Path to .mq5/.mqh/.ex5 to compile')
    me_c.add_argument('--syntax-only', action='store_true'); me_c.add_argument('--include', action='append')
    add_mt_args(me_c)
    me_c.set_defaults(func=cmd_code_compile)

    # ---- tester (top-level alias of mt tester) ----
    pt = sub.add_parser('tester', help='Strategy Tester operations')
    st = pt.add_subparsers(dest='tcmd', required=True)
    trun = st.add_parser('run', help='Run Strategy Tester via INI')
    trun.add_argument('--expert', required=True); trun.add_argument('--preset'); trun.add_argument('--symbol'); trun.add_argument('--period'); trun.add_argument('--visual', choices=['on','off'], default='off')
    add_mt_args(trun)
    # set_defaults moved below after function definition

    pci = smt.add_parser('chart', help='Operações de gráfico: indicator|expert|template')
    ci = pci.add_subparsers(dest='kind', required=True)
    cind = ci.add_parser('indicator', help='Indicadores no gráfico')
    cind_sub = cind.add_subparsers(dest='act', required=True)
    cind_a = cind_sub.add_parser('attach', help='Anexa indicador (usa defaults do projeto quando omitidos)')
    cind_a.add_argument('--indicator')
    cind_a.add_argument('--symbol')
    cind_a.add_argument('--period')
    cind_a.add_argument('--subwindow', type=int)
    add_mt_args(cind_a)
    cind_a.set_defaults(func=chart_indicator_attach)
    cind_d = cind_sub.add_parser('detach', help='Remove indicador')
    cind_d.add_argument('--indicator')
    cind_d.add_argument('--symbol')
    cind_d.add_argument('--period')
    cind_d.add_argument('--subwindow', type=int)
    add_mt_args(cind_d)
    cind_d.set_defaults(func=chart_indicator_detach)

    cexp = ci.add_parser('expert', help='Experts no gráfico')
    cexp_sub = cexp.add_subparsers(dest='act', required=True)
    cexp_a = cexp_sub.add_parser('attach', help='Anexa EA (suporta template)')
    cexp_a.add_argument('--expert', required=True)
    cexp_a.add_argument('--template')
    cexp_a.add_argument('--template-src')
    cexp_a.add_argument('--symbol')
    cexp_a.add_argument('--period')
    add_mt_args(cexp_a)
    cexp_a.set_defaults(func=chart_expert_attach)
    cexp_d = cexp_sub.add_parser('detach', help='Remove EA')
    cexp_d.add_argument('--symbol')
    cexp_d.add_argument('--period')
    add_mt_args(cexp_d)
    cexp_d.set_defaults(func=chart_expert_detach)

    ctpl = ci.add_parser('template', help='Templates do gráfico')
    ctpl_sub = ctpl.add_subparsers(dest='act', required=True)
    ctpl_a = ctpl_sub.add_parser('apply', help='Aplica template .tpl no gráfico')
    ctpl_a.add_argument('--template', required=True)
    ctpl_a.add_argument('--symbol')
    ctpl_a.add_argument('--period')
    add_mt_args(ctpl_a)
    ctpl_a.set_defaults(func=chart_template_apply)

    # mt tester run
    mtt = smt.add_parser('tester', help='Execução do Strategy Tester')
    mtt_sub = mtt.add_subparsers(dest='tcmd', required=True)
    mtt_run = mtt_sub.add_parser('run', help='Executa o tester com INI gerado automaticamente')
    mtt_run.add_argument('--expert', required=True, help='EA a testar (ex.: Pasta/MeuEA)')
    mtt_run.add_argument('--preset', help='Arquivo .set para parâmetros do EA')
    mtt_run.add_argument('--symbol', help='Símbolo (default: do projeto)')
    mtt_run.add_argument('--period', help='Timeframe (default: do projeto)')
    mtt_run.add_argument('--visual', choices=['on','off'], default='off')
    mtt_run.add_argument('--model', choices=['every-tick','open-prices','1-minute-ohlc','real-ticks'], default='every-tick')
    mtt_run.add_argument('--from', dest='date_from', help='Data inicial YYYY-MM-DD')
    mtt_run.add_argument('--to', dest='date_to', help='Data final YYYY-MM-DD')
    mtt_run.add_argument('--deposit', type=float, help='Depósito inicial (ex.: 10000)')
    mtt_run.add_argument('--currency', default='USD', help='Moeda do depósito (default: USD)')
    mtt_run.add_argument('--leverage', default='1:100', help='Alavancagem (ex.: 1:100)')
    mtt_run.add_argument('--report', help='Arquivo de relatório (html/xml). Default: ./tester_report.html')
    add_mt_args(mtt_run)
    def _fmt_date(d):
        if not d: return None
        try:
            return datetime.datetime.strptime(d, '%Y-%m-%d').strftime('%Y.%m.%d')
        except ValueError:
            raise SystemExit('Data inválida (use YYYY-MM-DD): ' + d)
    def cmd_mt_tester_run(args):
        with_project_defaults(args, use_indicator=False)
        terminal, _, data_dir = resolve_mt_context(args)
        # Monta INI do tester
        ini = Path.cwd() / 'tester.ini'
        sym = args.symbol
        per = timeframe_ok(args.period)
        vis = 1 if args.visual == 'on' else 0
        model_map = {
            'every-tick': 0,
            '1-minute-ohlc': 1,
            'open-prices': 2,
            'real-ticks': 3,
        }
        model = model_map.get(args.model, 0)
        lev = 100
        if isinstance(args.leverage, str) and ':' in args.leverage:
            try:
                lev = int(args.leverage.split(':',1)[1])
            except Exception:
                lev = 100
        rep = Path(args.report or (Path.cwd()/ 'tester_report.html'))
        lines = [
            '[Tester]',
            f'Expert={args.expert}',
            f'Symbol={sym}',
            f'Period={per}',
            f'Optimization=0',
            f'UseCloud=0',
            f'Visual={vis}',
            f'Model={model}',
            f'Deposit={(int(args.deposit) if args.deposit else 10000)}',
            f'Currency={args.currency}',
            f'Leverage={lev}',
            f'Report={to_windows_path(rep)}',
            f'ReplaceReport=1',
            f'Start=1',
        ]
        if args.preset:
            lines.append(f'ExpertParameters={to_windows_path(args.preset)}')
        fd = _fmt_date(args.date_from)
        td = _fmt_date(args.date_to)
        if fd:
            lines.append(f'FromDate={fd}')
        if td:
            lines.append(f'ToDate={td}')
        write_text_utf16(ini, "\n".join(lines)+"\n")
        # Inicia o Terminal
        extra = [f"/config:{to_windows_path(ini)}"]
        if getattr(args, 'profile', None):
            extra.append(f"/profile:{args.profile}")
        use_portable, dp_root = _should_use_portable(terminal, data_dir, getattr(args, 'portable', None))
        if use_portable:
            extra.append('/portable')
        else:
            extra.append(f"/datapath:{to_windows_path(dp_root)}")
        rc = run_win_exe(terminal, extra, detach=True)
        time.sleep(1.0)
        print_log_tail('tester run', data_dir=data_dir)
        print(tr('done'))
        return rc
    # bind after definition to avoid UnboundLocal errors
    mtt_run.set_defaults(func=cmd_mt_tester_run)
    trun.set_defaults(func=cmd_mt_tester_run)

    # mt logs tail
    mlogs = smt.add_parser('logs', help='Operações com logs do Terminal/engine')
    mlogs_sub = mlogs.add_subparsers(dest='logcmd', required=True)
    mlt = mlogs_sub.add_parser('tail', help='Tail rápido dos logs do projeto')
    mlt.add_argument('--lines', type=int, default=40)
    add_mt_args(mlt)
    mlt.set_defaults(func=cmd_mt_logs_tail)

    # mt create (verbo → objeto)
    mcreate = smt.add_parser('create', help='Criar artefatos relacionados ao MT5')
    mcreate_sub = mcreate.add_subparsers(dest='create_obj', required=True)
    mc_ini = mcreate_sub.add_parser('ini', help='Gera um .ini de inicialização do Terminal')
    mc_ini.add_argument('--symbol')
    mc_ini.add_argument('--period')
    mc_ini.add_argument('--expert', default='CommandListenerEA')
    mc_ini.add_argument('--expert-params', help='Arquivo .set para o Expert')
    mc_ini.add_argument('--template')
    mc_ini.add_argument('--script')
    mc_ini.add_argument('--script-params', help='Arquivo .set para o Script')
    mc_ini.add_argument('--shutdown', action='store_true', help='Encerrar o Terminal após script (quando aplicável)')
    mc_ini.add_argument('--output', default='listener.ini')
    add_mt_args(mc_ini)
    mc_ini.set_defaults(func=cmd_mt_ini_create)

    # Aliases de compatibilidade: mt ini create → mt create ini
    mini = smt.add_parser('ini', help='[LEGACY] use mt create ini')
    mini_sub = mini.add_subparsers(dest='ini_cmd', required=True)
    minic = mini_sub.add_parser('create', help='[LEGACY] use mt create ini')
    minic.add_argument('--symbol')
    minic.add_argument('--period')
    minic.add_argument('--expert', default='CommandListenerEA')
    minic.add_argument('--template')
    minic.add_argument('--script')
    minic.add_argument('--output', default='listener.ini')
    add_mt_args(minic)
    minic.set_defaults(func=cmd_mt_ini_create)

    # mt apply template
    mapply = smt.add_parser('apply', help='Aplicar operações no gráfico (template)')
    mapply_sub = mapply.add_subparsers(dest='apply_obj', required=True)
    ma_tpl = mapply_sub.add_parser('template', help='Aplica um template .tpl no gráfico')
    ma_tpl.add_argument('--template', required=True)
    ma_tpl.add_argument('--symbol')
    ma_tpl.add_argument('--period')
    add_mt_args(ma_tpl)
    ma_tpl.set_defaults(func=chart_template_apply)

    # mt install template/script
    def cmd_mt_install_template(args):
        terminal, metaeditor, data_dir = resolve_mt_context(args)
        templates_dir = win_to_wsl(data_dir) / 'MQL5' / 'Profiles' / 'Templates'
        ensure_dir(templates_dir)
        src = Path(args.file)
        if not src.exists():
            raise SystemExit(f"Arquivo .tpl não encontrado: {src}")
        dst = templates_dir / src.name
        shutil.copy2(src, dst)
        print(f"[install] template: {src.name} -> {to_windows_path(dst)}")
        time.sleep(0.2)
        print_log_tail('install template', data_dir=data_dir)
        print(tr('done'))
        return 0

    def cmd_mt_install_script(args):
        _, metaeditor, data_dir = resolve_mt_context(args)
        src = Path(args.file)
        if not src.exists():
            raise SystemExit(f"Arquivo de script não encontrado: {src}")
        install_file_to_mql(metaeditor, data_dir, f"Scripts/{src.name}", src, compile_after=not args.no_compile and src.suffix.lower()=='.mq5')
        time.sleep(0.2)
        print_log_tail('install script', data_dir=data_dir)
        print(tr('done'))
        return 0

    minst = smt.add_parser('install', help='Instala artefatos do MT5 no projeto')
    minst_sub = minst.add_subparsers(dest='inst_obj', required=True)
    minst_tpl = minst_sub.add_parser('template', help='Copia um .tpl para Profiles/\nTemplates do projeto')
    minst_tpl.add_argument('--file', required=True)
    add_mt_args(minst_tpl)
    minst_tpl.set_defaults(func=cmd_mt_install_template)
    minst_scr = minst_sub.add_parser('script', help='Instala script em MQL5/\nScripts (compila se .mq5)')
    minst_scr.add_argument('--file', required=True)
    minst_scr.add_argument('--no-compile', action='store_true')
    add_mt_args(minst_scr)
    minst_scr.set_defaults(func=cmd_mt_install_script)
    # mt attach/detach (unificados)
    ma = smt.add_parser('attach', help='Anexa indicador/EA/aplica template no gráfico (usa defaults do projeto quando omitidos)')
    g = ma.add_mutually_exclusive_group()
    g.add_argument('--indicator', help='Nome do indicador (default: do projeto)')
    g.add_argument('--expert', help='Nome do EA (Pasta/EA)')
    g.add_argument('--template', help='Nome do template .tpl')
    # expert extras
    ma.add_argument('--template-src', help='Arquivo .tpl para copiar antes de aplicar (EA)')
    ma.add_argument('--symbol')
    ma.add_argument('--period')
    ma.add_argument('--subwindow', type=int)
    add_mt_args(ma)
    def cmd_mt_attach(args):
        # template
        if getattr(args, 'template', None):
            return chart_template_apply(args)
        # expert
        if getattr(args, 'expert', None):
            return chart_expert_attach(args)
        # indicador (default)
        if not getattr(args, 'indicator', None):
            with_project_defaults(args, use_indicator=True)
            args.indicator = args.indicator
        return chart_indicator_attach(args)
    ma.set_defaults(func=cmd_mt_attach)

    md = smt.add_parser('detach', help='Remove indicador/EA do gráfico (usa defaults do projeto quando omitidos)')
    gd = md.add_mutually_exclusive_group()
    gd.add_argument('--indicator', help='Nome do indicador (default: do projeto)')
    gd.add_argument('--expert', help='Nome do EA (Pasta/EA)')
    md.add_argument('--symbol')
    md.add_argument('--period')
    md.add_argument('--subwindow', type=int)
    add_mt_args(md)
    def cmd_mt_detach(args):
        if getattr(args, 'expert', None):
            return chart_expert_detach(args)
        if not getattr(args, 'indicator', None):
            with_project_defaults(args, use_indicator=True)
        return chart_indicator_detach(args)
    md.set_defaults(func=cmd_mt_detach)

    # mt ini create (gera um INI para inicialização do Terminal)
    mini = smt.add_parser('ini', help='Operações com arquivos .ini do Terminal')
    mini_sub = mini.add_subparsers(dest='act', required=True)
    minic = mini_sub.add_parser('create', help='Gera um .ini com símbolo/período e expert/listener padrão')
    minic.add_argument('--symbol')
    minic.add_argument('--period')
    minic.add_argument('--expert', default='CommandListenerEA')
    minic.add_argument('--expert-params')
    minic.add_argument('--template')
    minic.add_argument('--script')
    minic.add_argument('--script-params')
    minic.add_argument('--shutdown', action='store_true')
    minic.add_argument('--output', default='listener.ini')
    add_mt_args(minic)
    minic.set_defaults(func=cmd_mt_ini_create)

    # Generate MQL (indicator/script) with connection handshake from YAML/preset

    pg = sub.add_parser('gen-mql', help=argparse.SUPPRESS)
    pg.add_argument('--type', choices=['indicator','script'], default='indicator', help='MQL output type')
    pg.add_argument('--name', help='Output name (without extension). Default: Conexao_ALGLIB')
    pg.add_argument('--yaml', help='YAML path with selection (ops: [...])')
    pg.add_argument('--preset', help='Optional preset label (metadata only)')
    pg.add_argument('--copy', action='store_true', help='Copy generated file to Terminal(s) MQL5 folder')
    pg.add_argument('--terminal-id', help='Specific Terminal ID to copy to (default: all)')
    pg.add_argument('--all', action='store_true', help='Copy to all Terminals')
    pg.add_argument('--no-from-exports', action='store_true', help='Do not scan dll_exports.cpp; use YAML/preset only')
    pg.add_argument('--mql5-root', help='Explicit MQL5 root (e.g. C:/mql5/MQL5) when copying generated MQL')
    pg.set_defaults(func=cmd_gen_mql)

    p = sub.add_parser('link', help=argparse.SUPPRESS)
    p.add_argument('--release', help='Explicit Release directory (default: build-win/<config>)')
    p.add_argument('--config', default='Release', help='Build configuration folder under build-win (default: Release)')
    p.add_argument('--libs', action='append', help='Explicit MQL5\\Libraries directory (repeatable). Overrides detection when present')
    p.add_argument('--project', help='Project id to use when no --libs is provided (default: last saved project)')
    p.add_argument('--agents', action='store_true', help='Also apply to all Tester Agent MQL5\\Libraries (C:/mql5/Tester/Agent-*/MQL5/Libraries)')
    p.set_defaults(func=cmd_link)

    # ---- project (save/list/use/show) ----
    proj = sub.add_parser('project', help='Manage saved projects (paths, presets, defaults)')
    proj_sub = proj.add_subparsers(dest='pcmd', required=True)
    def _cmd_proj_save(args):
        root = cli_root_default()
        data = load_projects(root)
        pid = args.id
        if not pid:
            print('ERROR: --id is required'); return 2
        now = datetime.datetime.now().isoformat(timespec='seconds')
        proj = data.get('projects', {}).get(pid, {})
        proj.update({
            'project': args.name or proj.get('project') or pid,
            'libs': args.libs or proj.get('libs',''),
            'metaeditor': args.metaeditor or proj.get('metaeditor',''),
            'terminal': args.terminal or proj.get('terminal',''),
            'data_dir': args.data_dir or proj.get('data_dir',''),
            'updated_at': now,
        })
        if 'created_at' not in proj:
            proj['created_at'] = now
        data.setdefault('projects', {})[pid] = proj
        if args.set_default:
            data['last_project'] = pid
        save_projects(root, data)
        print('Saved project:', pid)
        return 0
    ps = proj_sub.add_parser('save', help='Save or update a project')
    ps.add_argument('--id', required=True, help='Project id (e.g., legacy-wave)')
    ps.add_argument('--name', help='Preset/name for generator (e.g., GPU_LegacyWave1.0.4)')
    ps.add_argument('--libs', help='Path to MQL5\\Libraries for this project')
    ps.add_argument('--metaeditor', help='Path to metaeditor64.exe for this project')
    ps.add_argument('--terminal', help='Path to terminal64.exe for this project')
    ps.add_argument('--data-dir', help='Path to Data Folder (the MQL5 directory), e.g., C\\mql5\\MQL5')
    ps.add_argument('--set-default', action='store_true', help='Set as last_project')
    ps.set_defaults(func=_cmd_proj_save)

    def _cmd_proj_list(args):
        root = cli_root_default()
        data = load_projects(root)
        rows = []
        last = data.get('last_project')
        for pid, proj in data.get('projects', {}).items():
            mark = '*' if pid == last else ' '
            defs = proj.get('defaults', {})
            dsum = f"{defs.get('symbol','EURUSD')}/{defs.get('period','H1')} sub={defs.get('subwindow',1)} ind={defs.get('indicator','-') or '-'}"
            rows.append([mark+pid, proj.get('project',''), proj.get('libs',''), proj.get('data_dir',''), proj.get('terminal',''), proj.get('metaeditor',''), dsum, proj.get('updated_at','')])
        if rows:
            print_table(rows, headers=['ProjectID','Name','Libraries','DataFolder','Terminal','MetaEditor','Defaults','Updated'])
        else:
            print('No saved projects.')
        return 0
    pl = proj_sub.add_parser('list', help='List projects')
    pl.set_defaults(func=_cmd_proj_list)

    def _cmd_proj_use(args):
        root = cli_root_default()
        data = load_projects(root)
        if args.id not in data.get('projects', {}):
            print('ERROR: unknown project id:', args.id); return 2
        data['last_project'] = args.id
        save_projects(root, data)
        print('Now using project:', args.id)
        return 0
    pu = proj_sub.add_parser('use', help='Set default (last) project')
    pu.add_argument('--id', required=True)
    pu.set_defaults(func=_cmd_proj_use)

    def _cmd_proj_show(args):
        root = cli_root_default()
        data = load_projects(root)
        pid = args.id or data.get('last_project')
        proj = data.get('projects', {}).get(pid)
        if not proj:
            print('No project selected or not found.'); return 1
        defs = proj.get('defaults', {})
        dsum = f"{defs.get('symbol','EURUSD')}/{defs.get('period','H1')} sub={defs.get('subwindow',1)} ind={defs.get('indicator','-') or '-'}"
        rows = [[pid, proj.get('project',''), proj.get('libs',''), proj.get('data_dir',''), proj.get('terminal',''), proj.get('metaeditor',''), dsum, proj.get('updated_at','')]]
        print_table(rows, headers=['ProjectID','Name','Libraries','DataFolder','Terminal','MetaEditor','Defaults','Updated'])
        return 0
    psh = proj_sub.add_parser('show', help='Show current or specific project')
    psh.add_argument('--id', help='Project id (default: last)')
    psh.set_defaults(func=_cmd_proj_show)

    # project env/detect (realocação dos utilitários para dentro de project)
    p_env_proj = proj_sub.add_parser('env', help='Diagnóstico do ambiente (WSL/Windows, interop, python)')
    p_env_proj.set_defaults(func=cmd_env)
    p_detect_proj = proj_sub.add_parser('detect', help='Descobrir Terminals/MQL5 e bibliotecas (suporte ao init)')
    p_detect_proj.add_argument('--project', help='Project id to inspect (default: last saved project)')
    p_detect_proj.add_argument('--all', action='store_true', help='List every detected Terminal/Agent')
    p_detect_proj.set_defaults(func=cmd_detect)

    # defaults management: set/show per-project defaults (symbol, period, subwindow, indicator)
    def _cmd_proj_defaults_set(args):
        root = cli_root_default()
        data = load_projects(root)
        pid = args.id or data.get('last_project')
        if not pid or pid not in data.get('projects', {}):
            print('ERROR: selecione um projeto antes (mtcli project init/use)'); return 2
        proj = data['projects'][pid]
        defs = proj.get('defaults', {})
        changed = False
        if args.symbol:
            defs['symbol'] = args.symbol; changed = True
        if args.period:
            defs['period'] = args.period; changed = True
        if args.subwindow is not None:
            defs['subwindow'] = int(args.subwindow); changed = True
        if args.indicator:
            defs['indicator'] = args.indicator; changed = True
        if args.portable is not None:
            defs['portable'] = bool(int(args.portable)); changed = True
        if args.profile:
            defs['profile'] = args.profile; changed = True
        if changed:
            proj['defaults'] = defs
            proj['updated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
            save_projects(root, data)
        print('Defaults:', defs if defs else {'symbol':'EURUSD','period':'H1','subwindow':1,'indicator':'-'})
        return 0

    def _cmd_proj_defaults_show(args):
        root = cli_root_default()
        data = load_projects(root)
        pid = args.id or data.get('last_project')
        defs = resolve_project_defaults(root, pid)
        print('Defaults:', defs)
        return 0

    pdf = proj_sub.add_parser('defaults', help='Gerencia defaults de símbolo/período/subjanela/indicador do projeto')
    pdf_sub = pdf.add_subparsers(dest='dcmd', required=True)
    pdfs = pdf_sub.add_parser('set', help='Define defaults do projeto')
    pdfs.add_argument('--id', help='Project id (default: last)')
    pdfs.add_argument('--symbol')
    pdfs.add_argument('--period')
    pdfs.add_argument('--subwindow', type=int)
    pdfs.add_argument('--indicator')
    pdfs.add_argument('--portable', type=int, choices=[0,1], help='Forçar portable (1) ou não (0)')
    pdfs.add_argument('--profile', help='Profile do Terminal (ex.: Default)')
    pdfs.set_defaults(func=_cmd_proj_defaults_set)
    pdfsh = pdf_sub.add_parser('show', help='Mostra defaults vigentes')
    pdfsh.add_argument('--id', help='Project id (default: last)')
    pdfsh.set_defaults(func=_cmd_proj_defaults_show)

    # init: assistente integrado (detecta, salva, linka, instala listener e anexa indicador opcional)
    def _prompt(msg: str, default: str | None = None) -> str:
        if getattr(_cmd_proj_init, '_yes', False) and default is not None:
            print(f"{msg} {default}")
            return default
        v = input(f"{msg} ").strip()
        return v or (default or '')

    def _cmd_proj_init(args):
        _cmd_proj_init._yes = getattr(args, 'yes', False)
        root_cfg = cli_root_default()
        print('\n[init] Detectando Terminals…')
        detected = terminal_dirs()
        for i, d in enumerate(detected, 1):
            print(f" {i}. id={d['id']} libs={d['libs']}")
        print(f" {len(detected)+1}. Outro… (informar caminho do MQL5\\Libraries)")

        # Modo não interativo: escolher automaticamente o melhor candidato e NUNCA bloquear
        if getattr(args, 'yes', False):
            # Preferir /mnt/c/mql5 quando disponível
            preferred_libs = None
            for d in detected:
                libs = d.get('libs','')
                if libs.replace('\\\\','/').lower().endswith('/mql5/libraries') and ('/mnt/c/mql5' in libs.replace('\\\\','/').lower() or 'c:/mql5' in libs.replace('\\\\','/').lower()):
                    preferred_libs = libs
                    terminal = d
                    break
            if not preferred_libs:
                # fallback: primeiro detectado ou caminho manual padrão
                if detected:
                    terminal = detected[0]
                else:
                    terminal = {'id':'manual','root':'/mnt/c/mql5', 'libs':'/mnt/c/mql5/MQL5/Libraries'}
            pid = args.id or 'main-terminal'
            term_exe = '/mnt/c/mql5/terminal64.exe'
            meta_exe = '/mnt/c/mql5/metaeditor64.exe'
            data_dir = os.path.dirname(terminal['libs'])  # a própria pasta MQL5
        else:
            choice = _prompt(f"Selecione [1-{len(detected)+1}]:", str(1 if detected else len(detected)+1))
            try:
                idx = int(choice)
            except Exception:
                idx = len(detected)+1
            if idx==len(detected)+1:
                libs = _prompt('Caminho completo para MQL5\\Libraries:', '/mnt/c/mql5/MQL5/Libraries')
                if libs.startswith('~'):
                    libs = os.path.expanduser(libs)
                assert os.path.isdir(libs), f"Libraries inválido: {libs}"
                terminal = {'id':'manual','root': os.path.dirname(os.path.dirname(libs)), 'libs': libs}
            else:
                terminal = detected[idx-1]

            pid = args.id or _prompt('ID do projeto (ex.: main-terminal):', 'main-terminal')
            def _def(path1, path2):
                return path1 if os.path.isfile(path1) or os.path.isdir(path1) else path2
            term_default = _def('/mnt/c/mql5/terminal64.exe', '/mnt/c/mql5/terminal64.exe')
            meta_default = _def('/mnt/c/mql5/metaeditor64.exe', '/mnt/c/mql5/metaeditor64.exe')
            data_default = os.path.dirname(terminal['libs'])  # MQL5
            term_exe = _prompt(f"terminal64.exe [default: {term_default}]:", term_default)
            meta_exe = _prompt(f"metaeditor64.exe [default: {meta_default}]:", meta_default)
            data_dir = _prompt(f"Data Folder (pasta MQL5) [default: {data_default}]:", data_default)
        # Persistir projeto
        pdata = load_projects(root_cfg)
        now = datetime.datetime.now().isoformat(timespec='seconds')
        pdata.setdefault('projects', {}).setdefault(pid, {})
        pdata['projects'][pid].update({
            'project': args.name or pdata['projects'][pid].get('project') or pid,
            'libs': _normalize_path(terminal['libs']),
            'terminal': _normalize_path(term_exe),
            'metaeditor': _normalize_path(meta_exe),
            'data_dir': _normalize_path(data_dir),
            'updated_at': now,
        })
        pdata['last_project'] = pid
        save_projects(root_cfg, pdata)
        print(f"[init] Projeto salvo: {pid}")

        # Linkar MQL5\\Libraries com Release atual (silencioso se não existir)
        release_dir = os.path.join(root_cfg, 'build-win', 'Release')
        if os.path.isdir(release_dir):
            largs = argparse.Namespace(
                root=root_cfg,
                release=release_dir,
                config='Release',
                terminal_id=None,
                all=False,
                libs=[terminal['libs']],
                project=pid,
                agents=False,
            )
            cmd_link(largs)

        # Instalar listener e garantir execução em segundo plano — não bloqueante
        le = argparse.Namespace(project=pid, libs=None, terminal=None, metaeditor=None, data_dir=None, force=True, force_start=False)
        cmd_listener_ensure(le)

        # Anexar indicador opcionalmente (somente se informado explicitamente)
        if getattr(args, 'attach', False) and getattr(args, 'indicator', None):
            sym = (args.symbol or 'EURUSD'); tf = (args.period or 'H1'); sub = int(args.subwindow or 1)
            line = f"ATTACH_IND;{sym};{timeframe_ok(tf)};{args.indicator};{sub}"
            try:
                send_listener_command(Path(data_dir), line)
                print(f"[init] Comando agendado: {line}")
            except Exception as exc:
                print('[init] Aviso: não foi possível agendar ATTACH:', exc)
        time.sleep(0.8)
        print_log_tail('project init', data_dir=Path(data_dir))

        print('\n[init] Concluído. Projeto pronto. Use:')
        print(f"  mtcli attach --project {pid} --symbol EURUSD --period H1")
        print(f"  mtcli status --project {pid}")
        return 0

    pi = proj_sub.add_parser('init', help='Assistente: detectar, salvar, linkar e preparar MT5 (listener e indicador)')
    pi.add_argument('--id', help='Project id (default: main-terminal)')
    pi.add_argument('--name', help='Nome/preset exibido (default: id)')
    pi.add_argument('--yes', action='store_true', help='Aceitar defaults sem perguntar')
    pi.add_argument('--attach', action='store_true', help='Anexar indicador após preparar o projeto (requer --indicator)')
    pi.add_argument('--indicator', help='Nome do indicador para anexar durante o init')
    pi.add_argument('--symbol', help='Símbolo para anexar (default: EURUSD)')
    pi.add_argument('--period', help='Timeframe (default: H1)')
    pi.add_argument('--subwindow', type=int, help='Subjanela (default: 1)')
    pi.set_defaults(func=_cmd_proj_init)
    p = sub.add_parser('fetch-cuda', help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_fetch_cuda)

# ---- start (interactive wizard) ----
    p = sub.add_parser('start', help=argparse.SUPPRESS)
    p.add_argument('--project-id', help='Saved project id to use (default: last saved)')
    p.add_argument('--project', help='Project/preset name (default: GPU_LegacyWave1.0.4)')
    p.add_argument('--ea-name', default='CommandListener', help='EA name without extension (default: CommandListener)')
    p.add_argument('--libs', help='Target MQL5\\Libraries path (skips selection if provided)')
    p.add_argument('--metaeditor', help='Full path to metaeditor64.exe (skips auto-detect/prompt)')
    p.add_argument('--yes', action='store_true', help='Assume yes to prompts when safe')
    def cmd_start(args):
        # 1) Project name
        root_cfg = cli_root_default()
        pdata = load_projects(root_cfg)
        chosen_id = args.project_id or pdata.get('last_project')
        project = None
        default_name = 'GPU_LegacyWave1.0.4'
        if chosen_id and chosen_id in pdata.get('projects', {}):
            projinfo = pdata['projects'][chosen_id]
            project = projinfo.get('project') or default_name
            args.libs = args.libs or projinfo.get('libs')
            args.metaeditor = args.metaeditor or projinfo.get('metaeditor')
            print(f"Using saved project '{chosen_id}': name={project}")
        project = project or args.project or (input(f'Project name [{default_name}]: ').strip() or default_name)

        # 2) Pick Terminal (MQL5\Libraries)
        if args.libs:
            libs = args.libs
            if libs.startswith('~'):
                libs = os.path.expanduser(libs)
            if not os.path.isdir(libs):
                print('ERROR: invalid Libraries path:', libs)
                return 2
            terminal = {'id':'manual','root': os.path.dirname(os.path.dirname(libs)), 'libs': libs}
        else:
            detected = terminal_dirs()
            print('\nTerminals detected:')
            for i, d in enumerate(detected, 1):
                print(f" {i}. id={d['id']} libs={d['libs']}")
            print(f" {len(detected)+1}. Other... (enter a custom MQL5\\Libraries path)")
            choice = input(f'Select [1-{len(detected)+1}]: ').strip()
            try:
                idx = int(choice)
            except Exception:
                idx = len(detected)+1
            if idx==len(detected)+1:
                libs = input('Enter full path to MQL5\\Libraries: ').strip()
                if libs.startswith('~'):
                    libs = os.path.expanduser(libs)
                if not os.path.isdir(libs):
                    print('ERROR: invalid Libraries path. Aborting.')
                    return 2
                terminal = {'id':'manual','root': os.path.dirname(os.path.dirname(libs)), 'libs': libs}
            else:
                terminal = detected[idx-1]
        mql5_root = os.path.dirname(terminal['libs'])
        # Try to discover terminal executable for persistence
        term_exe = None
        for cand in ('/mnt/mql5/terminal64.exe', '/mnt/c/mql5/terminal64.exe', 'C:/mql5/terminal64.exe', 'C:/Program Files/MetaTrader 5/terminal64.exe'):
            if os.path.isfile(cand):
                term_exe = cand
                break
        if term_exe:
            terminal['exe'] = term_exe
        experts_dir = os.path.join(mql5_root, 'Experts', project)
        os.makedirs(experts_dir, exist_ok=True)

        # 3) Solicitar caminhos explícitos (Terminal/MetaEditor/Data Folder)
        projects_all = load_projects(root_cfg)
        last_id = projects_all.get('last_project')
        last_info = projects_all.get('projects', {}).get(last_id, {}) if last_id else {}
        term_default = last_info.get('terminal', '') or '/mnt/mql5/terminal64.exe'
        meta_default = last_info.get('metaeditor', '') or '/mnt/mql5/metaeditor64.exe'
        data_default = last_info.get('data_dir', os.path.dirname(os.path.dirname(terminal['libs'])))

        print('\nDefina os caminhos deste projeto (sem heurísticas):')
        if args.yes:
            t_in = term_default
            m_in = meta_default
            d_in = data_default
            print(f"  Terminal: {t_in}")
            print(f"  MetaEditor: {m_in}")
            print(f"  Data Folder: {d_in}")
        else:
            t_in = input(f"Terminal executable [default: {term_default}]: ").strip() or term_default
            m_in = input(f"MetaEditor executable [default: {meta_default}]: ").strip() or meta_default
            d_in = input(f"Data Folder raiz (pasta contendo MQL5) [default: {data_default}]: ").strip() or data_default
        if not os.path.isfile(_normalize_path(t_in)):
            print('ERROR: terminal64.exe inválido:', t_in); return 3
        if not os.path.isfile(_normalize_path(m_in)):
            print('ERROR: metaeditor64.exe inválido:', m_in); return 3
        if not os.path.isdir(_normalize_path(d_in)) or not os.path.isdir(os.path.join(_normalize_path(d_in), 'MQL5')):
            print('ERROR: Data Folder inválida (deve conter MQL5):', d_in); return 3
        term_exe = t_in
        meta = m_in

        # 4) Prepare and compile EA
        ea_base = args.ea_name
        ea_mq5 = os.path.join(experts_dir, ea_base + '.mq5')
        if not os.path.isfile(ea_mq5):
            # minimal EA template
            tpl = (
                '#property strict\n'
                f'// Auto-generated by mtcli start for project {project}\n'
                'int OnInit(){ Print("CommandListener init"); return(INIT_SUCCEEDED);}\n'
                'void OnTick(){}\n'
                'void OnChartEvent(const int id,const long &l,const double &d,const string &s){ PrintFormat("EVT %d %s", id, s); }\n'
            )
            with open(ea_mq5, 'w', encoding='utf-8') as f:
                f.write(tpl)
            print('Created EA:', ea_mq5)

        # compile EA
        log_dir = os.path.join(cli_root_default(), 'tools', 'mtcli_build')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'compile_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        # Use powershell for consistency
        ps = 'powershell.exe' if is_wsl() else 'powershell'
        meta_p = to_winpath(meta)
        log_p = to_winpath(log_path)
        ea_p = to_winpath(ea_mq5)
        cmd = [ps,'-NoProfile','-Command', f'& "{meta_p}" /log:"{log_p}" /compile:"{ea_p}"']
        run(cmd, check=False)
        ea_ex5 = os.path.join(experts_dir, ea_base + '.ex5')
        if not os.path.isfile(ea_ex5):
            print('ERROR: compile failed. See log:', log_path)
            return 4
        print('EA compiled OK:', ea_ex5)

        # create template and config
        tpl_dir = os.path.join(mql5_root, 'Profiles', 'Templates')
        os.makedirs(tpl_dir, exist_ok=True)
        tpl_path = os.path.join(tpl_dir, project + '.tpl')
        if not os.path.isfile(tpl_path):
            with open(tpl_path, 'w', encoding='utf-8') as f:
                f.write('; template placeholder for '+project+'\n')
        cfg_dir = os.path.join(terminal['root'], 'config')
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, f'terminal_{project}.ini')
        if not os.path.isfile(cfg_path):
            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.write('[Common]\nEnableAutoTrading=1\nAllowDllImport=1\n')
        print('Template and config prepared.')

        # 4) Generate preset via opgen
        root = cli_root_default()
        opgen = os.path.join(root, 'tools', 'bridge_gen', 'opgen.py')
        if os.path.isfile(opgen):
            run([sys.executable, opgen, '--root', root, '--preset', project, '--use-preset-selection'])
        else:
            print('WARN: opgen.py not found; skipping preset regeneration.')

        # 5) Build (skip in WSL; use existing DLL)
        if is_wsl():
            print('WSL environment: skipping CMake build (use existing dist-wave/alglib_bridge.dll).')
        else:
            bargs = argparse.Namespace(root=root, build_dir=None, arch='x64', config='Release', targets=['alglib_bridge'], parallel=os.cpu_count() or 8, preset=project, use_preset_selection=True, no_union_presets=False, extra=[])
            cmd_build(bargs)

        # 6) Link MQL5\\Libraries to build-win\\Release (no copies)
        largs = argparse.Namespace(
            root=root,
            release=os.path.join(root, 'build-win', 'Release'),
            config='Release',
            terminal_id=None,
            all=False,
            libs=[terminal['libs']],
            agents=False,
        )
        cmd_link(largs)

        # persist as last_project
        pdata = load_projects(root_cfg)
        # if we used a saved id, update it, else save under derived id
        pid = chosen_id or (project.lower().replace(' ', '-'))
        pdata.setdefault('projects', {}).setdefault(pid, {})
        # Persistir projeto com vinculação rígida
        pdata['projects'][pid].update({
            'project': project,
            'libs': terminal['libs'],
            'terminal': _normalize_path(term_exe),
            'metaeditor': _normalize_path(meta),
            'data_dir': _normalize_path(d_in),
            'updated_at': datetime.datetime.now().isoformat(timespec='seconds')
        })
        pdata['last_project'] = pid
        save_projects(root_cfg, pdata)
        print('\nStart sequence completed for project:', project, 'as id:', pid)
        return 0
    p.set_defaults(func=cmd_start)

    args = ap.parse_args()
    func = getattr(args, 'func', None)
    if not func:
        print_friendly_overview()
        return 0
    return func(args)


if __name__ == '__main__':
    sys.exit(main())
