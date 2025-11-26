import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

export function registerPingCommand(program: Command) {
  program
    .command('ping')
    .description('PING direto ao CommandListener')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendListenerCommand(info, 'PING', [], {
        timeoutMs: 4000,
        ensureRunning: true,
        allowRestart: false,
      });
      console.log(chalk.green('[ping] PING ok'));
    });
}
