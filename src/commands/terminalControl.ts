import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import {
  killTerminalIfRunning,
  killTerminalWindows,
  killTerminalByDatapath,
  startTerminalWindows,
  isTerminalRunning,
} from '../utils/commandListener.js';
import { collectAuthHints } from '../utils/logs.js';

const store = new ProjectStore();

export function registerTerminalControlCommands(program: Command) {
  const term = program.command('terminal').description('Controle do terminal MT5 (start/stop/status)');

  term
    .command('start')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal || !info.data_dir) {
        throw new Error('terminal/data_dir não configurados. Rode mtcli init ou activate antes.');
        }
      await killTerminalIfRunning(info.terminal).catch(() => {});
      await killTerminalWindows(info.terminal).catch(() => {});
      await killTerminalByDatapath(info.data_dir).catch(() => {});
      await startTerminalWindows(info.terminal, info.data_dir);
      await new Promise((r) => setTimeout(r, 6000));
      const running = await isTerminalRunning(info.terminal);
      const hints = collectAuthHints(info.data_dir, 20);
      if (hints.length) {
        console.log(chalk.gray('[terminal] Pistas de autenticação:'));
        console.log(hints.join('\n'));
      }
      if (!running) {
        console.log(chalk.yellow('[terminal] Processo não confirmado. Se a janela não estiver aberta, execute o run-terminal.ps1 no data_dir.'));
        return;
      }
      console.log(chalk.green('[terminal] Terminal iniciado.'));
    });

  term
    .command('stop')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal || !info.data_dir) {
        throw new Error('terminal/data_dir não configurados. Rode mtcli init ou activate antes.');
        }
      await killTerminalByDatapath(info.data_dir).catch(() => {});
      await killTerminalIfRunning(info.terminal).catch(() => {});
      await killTerminalWindows(info.terminal).catch(() => {});
      await new Promise((r) => setTimeout(r, 2000));
      const still = await isTerminalRunning(info.terminal);
      if (still) {
        console.log(chalk.yellow('[terminal] Processo ainda presente. Feche manualmente se necessário.'));
      } else {
        console.log(chalk.green('[terminal] Terminal encerrado.'));
      }
    });

  term
    .command('status')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal || !info.data_dir) {
        throw new Error('terminal/data_dir não configurados. Rode mtcli init ou activate antes.');
        }
      const running = await isTerminalRunning(info.terminal);
      console.log(`${running ? chalk.green('ON') : chalk.red('OFF')} ${info.terminal}`);
    });
}
