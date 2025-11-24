import path from 'path';
import fs from 'fs-extra';
import https from 'https';
import { pipeline } from 'stream/promises';
import { execa } from 'execa';
import { repoRoot } from '../config/projectStore.js';
import { promptYesNo } from './prompt.js';

const INSTALLER_URL =
  process.env.MTCLI_MT5_INSTALLER_URL || 'https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe';

// BASE_TERMINAL_DIR: onde guardamos todos os terminais baixados/instalados
const BASE_TERMINAL_DIR = process.env.MTCLI_BASE_TERMINAL_DIR || path.join(repoRoot(), 'projects', 'terminals', '_base');
const DOWNLOADS_ROOT = path.join(BASE_TERMINAL_DIR, '_downloads');
const FRESH_DIR = path.join(BASE_TERMINAL_DIR, 'mt5-fresh');
const INSTALLER_PATH = path.join(DOWNLOADS_ROOT, 'mt5setup.exe');

const TERMINAL_EXE = 'terminal64.exe';
const METAEDITOR_EXE = 'metaeditor64.exe';

function isWin(): boolean {
  return process.platform === 'win32' || !!process.env.WSL_DISTRO_NAME;
}

async function findPowerShell(): Promise<string | null> {
  const candidates = [
    process.env.POWERSHELL_EXE,
    'powershell.exe',
    '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
    'pwsh.exe',
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

async function isTerminalFolder(dir: string): Promise<boolean> {
  return fs.pathExists(path.join(dir, 'terminal64.exe'));
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
 * Instala um terminal dedicado para o projeto, copiando a pasta base informada.
 * Apenas a pasta é copiada; nenhuma configuração é alterada aqui.
 */
export async function installTerminalForProject(projectId: string, baseTerminalFolder?: string) {
  const baseRoot = await ensureBaseTerminal({ baseOverride: baseTerminalFolder });
  const destRoot = path.join(repoRoot(), 'projects', 'terminals', projectId);
  await fs.ensureDir(destRoot);
  if (path.resolve(baseRoot) !== path.resolve(destRoot)) {
    await fs.copy(baseRoot, destRoot, { overwrite: true });
  }

  // Pastas esperadas
  const dataDir = path.join(destRoot, 'MQL5');
  const libs = path.join(dataDir, 'Libraries');
  await fs.ensureDir(libs);

  const terminalExe = path.join(destRoot, 'terminal64.exe');
  const metaeditorExe = path.join(destRoot, 'MetaEditor64.exe');

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
  const installerPath = INSTALLER_PATH;

  if (await isTerminalFolder(target)) return target; // já baixado/instalado

  if (opts.interactive !== false) {
    const ok = await promptYesNo('Nenhum terminal base encontrado. Baixar e instalar o MetaTrader 5 agora?', true);
    if (!ok) {
      throw new Error('Download do MT5 cancelado pelo usuário.');
    }
  }

  console.log(`Baixando instalador MT5 de ${installerUrl}...`);
  await downloadFile(installerUrl, installerPath);

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
    await execa('cmd.exe', ['/C', `${installerWin} /auto ${pathArg}`], { stdio: 'inherit', windowsHide: true });
  }

  // Evita reinstalar: se instalou em pasta temporária do instalador, mova/copie
  if (!(await isTerminalFolder(target))) {
    // tenta localizar última instalação em Downloads padrão do instalador
    const alt = path.join(process.env['ProgramFiles'] || 'C:/Program Files', 'MetaTrader 5');
    if (await isTerminalFolder(alt)) {
      await fs.ensureDir(target);
      await fs.copy(alt, target, { overwrite: true });
    }
  }

  if (!(await isTerminalFolder(target))) {
    throw new Error('Instalação do MT5 falhou: terminal64.exe não encontrado.');
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
