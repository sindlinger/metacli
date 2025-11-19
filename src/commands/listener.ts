import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, repoRoot } from '../config/projectStore.js';
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

export async function runListenerInstance(options: ListenerRunOpts) {
  const info = await store.useOrThrow(options.project);
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
  if (!(await waitForTerminalStart())) {
    throw new Error('Terminal não permaneceu aberto após o restart. Verifique o listener.');
  }
  console.log(chalk.green('Terminal iniciado em segundo plano.'));
  await printLatestLogFromDataDir(dataDir);
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

async function killTerminalProcesses() {
  if (!platformIsWindows()) return;
  try {
    console.log(chalk.gray('[listener] encerrando instâncias atuais de terminal64.exe...'));
    await runCommand(
      powerShellExe(),
      ['-Command', 'Get-Process -Name terminal64 -ErrorAction SilentlyContinue | Stop-Process -Force'],
      { stdio: 'ignore' }
    );
  } catch {
    // ignora falhas (processo já fechado)
  }
}

async function ensureTerminalStopped(timeoutMs = 5000) {
  if (!platformIsWindows()) return;
  const start = Date.now();
  let attempted = false;
  while (await isListenerRunning()) {
    attempted = true;
    await killTerminalProcesses();
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
  await ensureTerminalStopped();
  await runListenerInstance(options);
}

export async function isListenerRunning(): Promise<boolean> {
  if (!platformIsWindows()) return true;
  try {
    await execa(
      powerShellExe(),
      ['-Command', 'if (Get-Process -Name terminal64 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }'],
      { stdio: 'ignore' }
    );
    return true;
  } catch {
    return false;
  }
}

async function waitForTerminalStart(timeoutMs = 5000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await isListenerRunning()) return true;
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

export function registerListenerCommands(program: Command) {
  const listener = program.command('listener').description('Opera o CommandListenerEA/terminal');

  listener
    .command('start')
    .description('Abre o terminal com o listener.ini')
    .option('--project <id>', 'Projeto configurado')
    .option('--config <path>', 'Arquivo listener.ini customizado')
    .option('--profile <name>', 'Perfil do MT5', 'Default')
    .action(async (opts) => {
      await runListenerInstance(opts);
    });

  listener
    .command('restart')
    .description('Fecha o terminal atual e inicia novamente com listener.ini')
    .option('--project <id>')
    .option('--config <path>')
    .option('--profile <name>')
    .action(async (opts) => restartListenerInstance(opts));

  listener
    .command('status')
    .description('Mostra o log mais recente do listener')
    .option('--project <id>')
    .action(async (opts) => {
      await showListenerStatus(opts.project);
    });
}
