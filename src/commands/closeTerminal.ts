import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import {
  killTerminalIfRunning,
  killTerminalWindows,
  killTerminalByDatapath,
  isTerminalRunning,
} from '../utils/commandListener.js';

const store = new ProjectStore();

export function registerCloseTerminalCommand(program: Command) {
  program
    .command('close-terminal')
    .description('Encerra o terminal do projeto (somente o datapath atual)')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal || !info.data_dir) {
        throw new Error('terminal/data_dir não configurados. Rode mtcli init primeiro.');
      }
      await killTerminalByDatapath(info.data_dir).catch(() => {});
      await killTerminalIfRunning(info.terminal).catch(() => {});
      await killTerminalWindows(info.terminal).catch(() => {});
      await new Promise((r) => setTimeout(r, 2000));
      const still = await isTerminalRunning(info.terminal);
      if (still) {
        console.log(
          chalk.yellow(
            '[close-terminal] Processo ainda presente. Feche manualmente se necessário.'
          )
        );
      } else {
        console.log(chalk.green('[close-terminal] Terminal encerrado.'));
      }
    });
}
