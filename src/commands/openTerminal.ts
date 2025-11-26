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

export function registerOpenTerminalCommand(program: Command) {
  program
    .command('open-terminal')
    .description('Abre o terminal do projeto (mata só o do projeto e relança com /portable /datapath)')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal || !info.data_dir) {
        throw new Error('terminal/data_dir não configurados. Rode mtcli init primeiro.');
      }
      const terminal = info.terminal;
      const dataDir = info.data_dir;

      await killTerminalIfRunning(terminal).catch(() => {});
      await killTerminalWindows(terminal).catch(() => {});
      await killTerminalByDatapath(dataDir).catch(() => {});
      await startTerminalWindows(terminal, dataDir);
      // aguarda subir e valida
      await new Promise((r) => setTimeout(r, 7000));
      const running = await isTerminalRunning(terminal);
      const authHints = collectAuthHints(dataDir, 40);
      if (authHints.length) {
        console.log(chalk.gray('[open-terminal] Pistas de autenticação nos logs:'));
        console.log(authHints.join('\n'));
      }
      if (!running) {
        console.log(
          chalk.yellow(
            '[open-terminal] Processo não confirmado após o start. Se a janela não estiver aberta, execute o run-terminal.ps1 no data_dir.'
          )
        );
        return;
      }
      console.log(chalk.green('[open-terminal] Terminal do projeto iniciado.'));
    });
}
