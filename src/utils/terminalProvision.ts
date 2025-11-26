import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { repoRoot } from '../config/projectStore.js';

const DEFAULT_BASE = path.join(
  repoRoot(),
  'projects',
  'terminals',
  'project-72D7079820AB4E374CDC07CD933C3265'
);

function resolveBase(): string {
  if (process.env.MTCLI_BASE_TERMINAL_DIR && process.env.MTCLI_BASE_TERMINAL_DIR.trim()) {
    return path.resolve(process.env.MTCLI_BASE_TERMINAL_DIR);
  }
  return DEFAULT_BASE;
}

const BIN_FILES = ['terminal64.exe', 'MetaEditor64.exe', 'metatester64.exe', 'Terminal.ico'];
const LISTENER_FILES = [
  path.join('MQL5', 'Experts', 'CommandListenerEA.mq5'),
  path.join('MQL5', 'Experts', 'CommandListenerEA.ex5'),
];
const CONFIG_EXTRAS = ['servers.dat', 'accounts.dat'];

function resolveBaseConfigDir(base: string): string {
  const upper = path.join(base, 'Config');
  const lower = path.join(base, 'config');
  if (fs.pathExistsSync(upper)) return upper;
  if (fs.pathExistsSync(lower)) return lower;
  return upper; // fallback
}

/**
 * Copia apenas os executáveis a partir de um terminal base (project-72... ou MTCLI_BASE_TERMINAL_DIR).
 * Se já existir terminal64.exe em dataDir, não faz nada.
 */
export async function provisionTerminalFromBase(dataDir: string): Promise<void> {
  const terminalPath = path.join(dataDir, 'terminal64.exe');
  const base = resolveBase();
  if (!(await fs.pathExists(base))) {
    throw new Error(`Terminal base não encontrado em ${base}`);
  }
  const baseConfigDir = resolveBaseConfigDir(base);
  await fs.ensureDir(dataDir);
  // copia executáveis se faltarem
  for (const f of BIN_FILES) {
    const dst = path.join(dataDir, f);
    if (await fs.pathExists(dst)) continue;
    const src = path.join(base, f);
    if (!(await fs.pathExists(src))) continue;
    await fs.copy(src, dst, { overwrite: true });
  }
  // Copia o EA do CommandListener
  for (const f of LISTENER_FILES) {
    const src = path.join(base, f);
    const dst = path.join(dataDir, f);
    if (await fs.pathExists(src)) {
      await fs.ensureDir(path.dirname(dst));
      await fs.copy(src, dst, { overwrite: true });
    }
  }
  // Cria apenas as pastas vazias essenciais; conteúdos virão de factory/deploy depois.
  const dirs = ['MQL5', 'Profiles', 'Tester', 'Logs', 'Config', 'Templates', 'Scripts', 'Experts'];
  for (const d of dirs) {
    const dstDir = path.join(dataDir, d);
    await fs.ensureDir(dstDir);
  }
  // Copia arquivos de servidor necessários para login
  for (const fname of CONFIG_EXTRAS) {
    const src = path.join(baseConfigDir, fname);
    const dst = path.join(dataDir, 'Config', fname);
    if (await fs.pathExists(src)) {
      await fs.copy(src, dst, { overwrite: false });
    }
  }
  // origin.txt registra a pasta do usuário/terminal
  const originFile = path.join(dataDir, 'origin.txt');
  if (!(await fs.pathExists(originFile))) {
    await fs.writeFile(originFile, path.resolve(dataDir), 'utf8');
  }
  console.log(chalk.green(`[init] Terminal provisionado a partir de base: ${base}`));
}
