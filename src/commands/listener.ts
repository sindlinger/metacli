import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, ProjectInfo, repoRoot } from '../config/projectStore.js';
import { runCommand } from '../utils/shell.js';
import { toWinPath, toWslPath, normalizePath, platformIsWindows, resolvePowerShell } from '../utils/paths.js';
import { printLatestLogFromDataDir } from '../utils/logs.js';
import { execa } from 'execa';

const store = new ProjectStore();

export interface ListenerRunOpts {
  project?: string;
  config?: string;
  profile?: string;
}

function psQuoteLiteral(input: string): string {
  const escaped = input.replace(/'/g, "''");
  return `'${escaped}'`;
}

async function runListenerForInfo(info: ProjectInfo, options: ListenerRunOpts) {
  if (!info.terminal) {
    throw new Error('terminal64.exe não configurado no projeto');
  }
  const dataDir = info.data_dir;
  if (!dataDir) {
    throw new Error('Defina data_dir no projeto (mtcli project save --data-dir ...)');
  }
  const configPath = normalizePath(options.config || path.join(repoRoot(), 'listener.ini'));
  if (!(await fs.pathExists(configPath))) {
    throw new Error(`listener.ini não encontrado: ${configPath}`);
  }
  const args = [
    `/config:${toWinPath(configPath)}`,
    `/profile:${options.profile || 'Default'}`,
    `/datapath:${toWinPath(dataDir)}`,
  ];
  const exe = toWslPath(info.terminal);
  console.log(chalk.gray(`[listener] ${exe} ${args.join(' ')}`));
  await runCommand(exe, args, { detach: true, stdio: 'ignore' });
  if (!(await waitForTerminalStart(info.terminal))) {
    throw new Error('Terminal não permaneceu aberto após o restart. Verifique o listener.');
  }
  console.log(chalk.green('Terminal iniciado em segundo plano.'));
  await printLatestLogFromDataDir(dataDir);
}

export async function runListenerInstance(options: ListenerRunOpts) {
  const info = await store.useOrThrow(options.project);
  await runListenerForInfo(info, options);
}

let cachedPowerShell: string | null = null;

function powerShellExe(): string {
  if (!platformIsWindows()) {
    throw new Error('Operação disponível apenas no Windows/WSL.');
  }
  if (!cachedPowerShell) {
    cachedPowerShell = resolvePowerShell();
  }
  return cachedPowerShell;
}

async function killTerminalProcesses(targetExe?: string) {
  if (!platformIsWindows()) return;
  try {
    const script = targetExe
      ? `
$path = ${psQuoteLiteral(toWinPath(targetExe))};
$procs = Get-Process -Name terminal64 -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $path };
if ($procs) { $procs | Stop-Process -Force }
`
      : 'Get-Process -Name terminal64 -ErrorAction SilentlyContinue | Stop-Process -Force';
    console.log(chalk.gray('[listener] encerrando instâncias atuais do terminal configurado...'));
    await runCommand(powerShellExe(), ['-NoLogo', '-Command', script], { stdio: 'ignore' });
  } catch {
    // ignora falhas (processo já fechado)
  }
}

async function ensureTerminalStopped(targetExe?: string, timeoutMs = 5000) {
  if (!platformIsWindows()) return;
  const start = Date.now();
  let attempted = false;
  while (await isListenerRunning(targetExe)) {
    attempted = true;
    await killTerminalProcesses(targetExe);
    if (Date.now() - start >= timeoutMs) {
      throw new Error('Não foi possível encerrar terminal64.exe automaticamente. Feche manualmente e tente novamente.');
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (attempted) {
    console.log(chalk.green('[listener] terminal anterior encerrado.'));
  }
}

export async function restartListenerInstance(options: ListenerRunOpts) {
  const info = await store.useOrThrow(options.project);
  await ensureTerminalStopped(info.terminal);
  await runListenerForInfo(info, options);
}

export async function isListenerRunning(targetExe?: string): Promise<boolean> {
  if (!platformIsWindows()) return true;
  try {
    const args = targetExe
      ? [
          '-NoLogo',
          '-Command',
          `
$path = ${psQuoteLiteral(toWinPath(targetExe))};
if (Get-Process -Name terminal64 -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $path }) { exit 0 } else { exit 1 }
`,
        ]
      : ['-NoLogo', '-Command', 'if (Get-Process -Name terminal64 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }'];
    await execa(powerShellExe(), args, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

async function waitForTerminalStart(targetExe?: string, timeoutMs = 5000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await isListenerRunning(targetExe)) return true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}


async function showListenerStatus(project?: string) {
  const info = await store.useOrThrow(project);
  const dataDir = info.data_dir;
  if (!dataDir) {
    throw new Error('data_dir não configurado para este projeto.');
  }
  await printLatestLogFromDataDir(dataDir);
}

function warnLegacy() {
  console.log(
    chalk.yellow('[listener] Este comando é legado. Use `mtcli init` para controlar o ambiente sempre que possível.')
  );
}

export function registerListenerCommands(program: Command) {
  const listener = program
    .command('listener')
    .description('[LEGACY] Controle manual do CommandListenerEA (prefira `mtcli init`)');

  listener
    .command('start')
    .description('Abre o terminal com o listener.ini')
    .option('--project <id>', 'Projeto configurado')
    .option('--config <path>', 'Arquivo listener.ini customizado')
    .option('--profile <name>', 'Perfil do MT5', 'Default')
    .action(async (opts) => {
      warnLegacy();
      await runListenerInstance(opts);
    });

  listener
    .command('restart')
    .description('Fecha o terminal atual e inicia novamente com listener.ini')
    .option('--project <id>')
    .option('--config <path>')
    .option('--profile <name>')
    .action(async (opts) => {
      warnLegacy();
      await restartListenerInstance(opts);
    });

  listener
    .command('status')
    .description('Mostra o log mais recente do listener')
    .option('--project <id>')
    .action(async (opts) => {
      warnLegacy();
      await showListenerStatus(opts.project);
    });
}
