import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

export function registerRawCommands(program: Command) {
  program
    .command('listener')
    .description('Envia comando bruto ao CommandListener (debug)')
    .requiredOption('--type <TYPE>', 'Tipo, ex.: PING')
    .option('--params <p1,p2,...>', 'Parâmetros separados por vírgula')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const params = opts.params ? opts.params.split(',').map((p: string) => p.trim()) : [];
      const resp = await sendListenerCommand(info, opts.type, params, { timeoutMs: 10000 });
      console.log(chalk.cyan(`[listener] ${resp.status}: ${resp.message}`));
      if (resp.data.length) {
        resp.data.forEach((line) => console.log('  ' + line));
      }
    });
}
