import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, repoRoot } from '../config/projectStore.js';
import { runCommand } from '../utils/shell.js';
import { toWinPath, toWslPath, normalizePath } from '../utils/paths.js';
import { tailFile } from '../utils/logs.js';

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
  console.log(chalk.green('Terminal iniciado em segundo plano.'));
}

async function killTerminalProcesses() {
  try {
    await runCommand('powershell.exe', ['-Command', 'Get-Process terminal64 -ErrorAction SilentlyContinue | Stop-Process -Force'], { stdio: 'ignore' });
  } catch {
    // ignora falhas (processo já fechado)
  }
}

export async function restartListenerInstance(options: ListenerRunOpts) {
  await killTerminalProcesses();
  await runListenerInstance(options);
}

async function showListenerStatus(project?: string) {
  const info = await store.useOrThrow(project);
  const dataDir = info.data_dir;
  if (!dataDir) {
    throw new Error('data_dir não configurado para este projeto.');
  }
  const logDir = path.join(dataDir, 'MQL5', 'Logs');
  if (!(await fs.pathExists(logDir))) {
    console.log('Pasta de logs não existe ainda. Abra o MT5 pelo menos uma vez.');
    return;
  }
  const files = (await fs.readdir(logDir)).filter((name) => name.endsWith('.log'));
  if (files.length === 0) {
    console.log('Nenhum log encontrado em', logDir);
    return;
  }
  files.sort((a, b) => {
    const aStat = fs.statSync(path.join(logDir, a));
    const bStat = fs.statSync(path.join(logDir, b));
    return bStat.mtimeMs - aStat.mtimeMs;
  });
  const latest = path.join(logDir, files[0]);
  console.log(chalk.bold(`Arquivo: ${files[0]}`));
  console.log(tailFile(latest));
}

export function registerListenerCommands(program: Command) {
  const listener = program.command('listener').description('Opera o CommandListenerEA/terminal');

  listener
    .command('run')
    .description('Abre o terminal com o listener.ini')
    .option('--project <id>', 'Projeto configurado')
    .option('--config <path>', 'Arquivo listener.ini customizado')
    .option('--profile <name>', 'Perfil do MT5', 'Default')
    .action(async (opts) => {
      await runListenerInstance(opts);
    });

  listener
    .command('ensure')
    .description('Alias para listener run')
    .option('--project <id>')
    .option('--config <path>')
    .option('--profile <name>')
    .action(async (opts) => runListenerInstance(opts));

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
