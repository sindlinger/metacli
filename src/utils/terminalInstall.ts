import path from 'path';
import fs from 'fs-extra';
import https from 'https';
import { pipeline } from 'stream/promises';
import { execa } from 'execa';
import { repoRoot } from '../config/projectStore.js';
import { promptYesNo } from './prompt.js';

const INSTALLER_URL =
  process.env.MTCLI_MT5_INSTALLER_URL || 'https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe';
const LOCAL_INSTALLER = path.join(repoRoot(), 'factory', 'dukascopy5setup.exe');

// BASE_TERMINAL_DIR: onde guardamos todos os terminais baixados/instalados
const BASE_TERMINAL_DIR = process.env.MTCLI_BASE_TERMINAL_DIR || path.join(repoRoot(), 'projects', 'terminals', '_base');
const DOWNLOADS_ROOT = path.join(BASE_TERMINAL_DIR, '_downloads');
const FRESH_DIR = path.join(BASE_TERMINAL_DIR, 'mt5-fresh');
const INSTALLER_PATH = path.join(DOWNLOADS_ROOT, 'mt5setup.exe');
const POWERSHELL_PORTABLE_DIR = path.join(BASE_TERMINAL_DIR, '_tools', 'powershell');

const TERMINAL_EXE = 'terminal64.exe';
const METAEDITOR_EXE = 'metaeditor64.exe';
const APPDATA_ENV = 'APPDATA';

function isWin(): boolean {
  return process.platform === 'win32' || !!process.env.WSL_DISTRO_NAME;
}

function appendToPathIfMissing(dir: string) {
  if (!dir) return;
  const pathEnv = process.env.PATH || '';
  if (!pathEnv.split(path.delimiter).includes(dir)) {
    process.env.PATH = `${dir}${path.delimiter}${pathEnv}`;
  }
}

async function findPowerShell(): Promise<string | null> {
  const candidates = [
    process.env.POWERSHELL_EXE,
    'powershell.exe',
    '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
    'pwsh.exe',
    path.join(POWERSHELL_PORTABLE_DIR, 'pwsh.exe'),
  ].filter(Boolean) as string[];
  for (const c of candidates) {
    try {
      await execa(c, ['-NoProfile', '-Command', '$PSVersionTable.PSVersion'], { stdout: 'ignore', stderr: 'ignore' });
      return c;
    } catch {
      // ignore
    }
  }
  return null;
}

async function toWindowsPath(p: string): Promise<string> {
  if (process.platform !== 'linux' || !process.env.WSL_DISTRO_NAME) return p;
  const { stdout } = await execa('wslpath', ['-w', p]);
  return stdout.trim();
}

async function getWinAppData(): Promise<string> {
  if (process.env[APPDATA_ENV]) return process.env[APPDATA_ENV] as string;
  const { stdout } = await execa('cmd.exe', ['/C', 'echo', `%${APPDATA_ENV}%`]);
  return stdout.trim();
}

async function findNewestDataDir(): Promise<string | null> {
  const appData = await getWinAppData();
  const terminalsRoot = path.join(appData, 'MetaQuotes', 'Terminal');
  if (!(await fs.pathExists(terminalsRoot))) return null;
  const entries = await fs.readdir(terminalsRoot).catch(() => []);
  let newest: { dir: string; mtime: number } | null = null;
  for (const e of entries) {
    const dir = path.join(terminalsRoot, e);
    const stat = await fs.stat(dir).catch(() => null);
    if (!stat || !stat.isDirectory()) continue;
    if (!newest || stat.mtimeMs > newest.mtime) {
      newest = { dir, mtime: stat.mtimeMs };
    }
  }
  return newest?.dir || null;
}

async function rewriteOrigin(dataDir: string, installRoot: string): Promise<void> {
  const originFile = path.join(dataDir, 'origin.txt');
  const win = normalizeWinPath(await toWindowsPath(installRoot));
  await fs.writeFile(originFile, win, 'utf8');
}

async function isTerminalFolder(dir: string): Promise<boolean> {
  return fs.pathExists(path.join(dir, 'terminal64.exe'));
}

function normalizeWinPath(p: string): string {
  return p.replace(/[\\/]+/g, '\\').replace(/\\+$/g, '').toLowerCase();
}

async function findDataDirByOrigin(installRoot: string): Promise<string | null> {
  const appData = await getWinAppData();
  const terminalsRoot = path.join(appData, 'MetaQuotes', 'Terminal');
  if (!(await fs.pathExists(terminalsRoot))) return null;
  const entries = await fs.readdir(terminalsRoot);
  const installWin = normalizeWinPath(await toWindowsPath(installRoot));
  for (const e of entries) {
    const originPath = path.join(terminalsRoot, e, 'origin.txt');
    try {
      const content = normalizeWinPath((await fs.readFile(originPath, 'utf8')).trim());
      if (content === installWin) {
        return path.join(terminalsRoot, e);
      }
    } catch {
      // ignore
    }
  }
  // fallback: usa o diretório mais recente se não encontrar origin
  let newest: { dir: string; mtime: number } | null = null;
  for (const e of entries) {
    const dir = path.join(terminalsRoot, e);
    const stat = await fs.stat(dir).catch(() => null);
    if (!stat || !stat.isDirectory()) continue;
    if (!newest || stat.mtimeMs > newest.mtime) {
      newest = { dir, mtime: stat.mtimeMs };
    }
  }
  return newest?.dir || null;
}

async function waitForDataDir(installRoot: string, timeoutMs = 30000): Promise<string | null> {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const found = await findNewestDataDir();
    if (found) return found;
    const elapsed = Date.now() - started;
    if (elapsed > 0 && elapsed % 5000 < 1000) {
      const sec = Math.floor(elapsed / 1000);
      console.log(`[install] aguardando data_dir aparecer... ${sec}s`);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return null;
}

async function touchDataDirByMetaEditor(installRoot: string): Promise<void> {
  const meta = path.join(installRoot, METAEDITOR_EXE);
  const metaWin = await toWindowsPath(meta);
  await execa('cmd.exe', ['/C', `${metaWin} /version`], { timeout: 30000, windowsHide: true }).catch(() => {});
  await new Promise((res) => setTimeout(res, 2000));
}


async function downloadFile(url: string, destination: string): Promise<void> {
  await fs.ensureDir(path.dirname(destination));
  return new Promise((resolve, reject) => {
    const doRequest = (current: string, redirects = 0) => {
      https
        .get(current, (res) => {
          if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            if (redirects > 5) {
              reject(new Error('Muitas redireções ao baixar o instalador MT5.'));
              return;
            }
            const next = res.headers.location.startsWith('http')
              ? res.headers.location
              : new URL(res.headers.location, current).toString();
            res.destroy();
            doRequest(next, redirects + 1);
            return;
          }
          if (res.statusCode !== 200) {
            reject(new Error(`Falha ao baixar MT5 (status ${res.statusCode})`));
            return;
          }
          const file = fs.createWriteStream(destination);
          pipeline(res, file)
            .then(() => resolve())
            .catch((err) => reject(err));
        })
        .on('error', reject);
    };
    doRequest(url);
  });
}

/**
 * Instala um terminal dedicado para o projeto, sempre criando/garantindo
 * uma instalação própria em projects/terminals/<projectId>.
 * Não reutiliza pasta externa; baixa/instala se necessário.
 */
interface InstallOptions {
  skipInstall?: boolean;
}

export async function installTerminalForProject(projectId: string, options: InstallOptions = {}) {
  const destRoot = path.join(repoRoot(), 'projects', 'terminals', projectId);
  await fs.ensureDir(destRoot);

  // Se já houver terminal instalado ali, reaproveita; caso contrário, instala novo (a menos que skipInstall=true).
  if (!(await isTerminalFolder(destRoot))) {
    if (options.skipInstall) {
      throw new Error(`skipInstall solicitado, mas não há terminal em ${destRoot}`);
    }
    console.log(`[install] iniciando instalação MT5 em ${destRoot}`);
    await downloadFreshTerminal({ targetDir: destRoot, interactive: false });
  } else {
    console.log(`[install] terminal já existe em ${destRoot}, reutilizando.`);
  }

  const terminalExe = path.join(destRoot, 'terminal64.exe');
  const metaeditorExe = path.join(destRoot, 'MetaEditor64.exe');
  // data_dir em modo normal: localizar via origin.txt em %APPDATA%
  let dataDir = await waitForDataDir(destRoot, 30000);
  if (!dataDir) {
    console.log('[install] tocando MetaEditor para forçar criação de data_dir');
    await touchDataDirByMetaEditor(destRoot);
    dataDir = await waitForDataDir(destRoot, 30000);
  }
  if (!dataDir) {
    throw new Error('Não foi possível localizar o data_dir gerado (origin.txt) para esta instalação. Abra o terminal manualmente uma vez e tente novamente.');
  }

  const libs = path.join(dataDir, 'MQL5', 'Libraries');
  await fs.ensureDir(libs);

  return {
    terminal: terminalExe,
    metaeditor: metaeditorExe,
    dataDir,
    libs,
    root: destRoot,
  };
}

interface DownloadOpts {
  targetDir?: string;
  installerUrl?: string;
  interactive?: boolean;
}

export async function downloadFreshTerminal(opts: DownloadOpts = {}): Promise<string> {
  if (!isWin()) {
    throw new Error('Download automático do MT5 requer Windows ou WSL. Informe MTCLI_BASE_TERMINAL se estiver em outra plataforma.');
  }

  const target = path.resolve(opts.targetDir || FRESH_DIR);
  const installerUrl = opts.installerUrl || INSTALLER_URL;
  // prefere instalador local (factory/dukascopy5setup.exe) se existir
  const installerPath = (await fs.pathExists(LOCAL_INSTALLER)) ? LOCAL_INSTALLER : INSTALLER_PATH;

  if (await isTerminalFolder(target)) return target; // já instalado

  if (!(await fs.pathExists(installerPath))) {
    if (opts.interactive !== false) {
      const ok = await promptYesNo('Nenhum terminal base encontrado. Baixar e instalar o MetaTrader 5 agora?', true);
      if (!ok) {
        throw new Error('Download do MT5 cancelado pelo usuário.');
      }
    }
    console.log(`Baixando instalador MT5 de ${installerUrl}...`);
    await downloadFile(installerUrl, installerPath);
  }

  const targetWin = await toWindowsPath(target);
  const installerWin = await toWindowsPath(installerPath);
  await fs.ensureDir(target);

  console.log(`Instalando MT5 em ${targetWin} (modo automático)...`);
  const pathArg = `/path:"${targetWin}"`;
  const ps = await findPowerShell();
  if (ps) {
    await execa(ps, ['-NoProfile', '-Command', `& "${installerWin}" /auto ${pathArg}`], {
      stdio: 'inherit',
      windowsHide: true,
    });
  } else {
    await execa('cmd.exe', ['/C', `"${installerWin}" /auto ${pathArg}`], { stdio: 'inherit', windowsHide: true });
  }

  for (let i = 0; i < 180; i += 1) {
    if (await isTerminalFolder(target)) break;
    if (i % 5 === 0 && i > 0) {
      console.log(`[install] aguardando instalador concluir... ${i}s`);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }

  if (!(await isTerminalFolder(target))) {
    throw new Error('Instalação do MT5 falhou: terminal64.exe não encontrado no destino solicitado.');
  }

  return target;
}

export async function ensureBaseTerminal({ baseOverride }: { baseOverride?: string } = {}): Promise<string> {
  if (baseOverride) {
    const resolved = path.resolve(baseOverride);
    if (!(await isTerminalFolder(resolved))) {
      throw new Error(`Terminal base não encontrado em ${resolved}`);
    }
    return resolved;
  }

  const envBase = process.env.MTCLI_BASE_TERMINAL;
  if (envBase) {
    const resolved = path.resolve(envBase);
    if (!(await isTerminalFolder(resolved))) {
      throw new Error(`MTCLI_BASE_TERMINAL aponta para ${resolved}, mas terminal64.exe não foi encontrado.`);
    }
    return resolved;
  }

  if (await isTerminalFolder(FRESH_DIR)) {
    return FRESH_DIR;
  }

  return downloadFreshTerminal({ targetDir: FRESH_DIR, interactive: true });
}
