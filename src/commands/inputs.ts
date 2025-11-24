import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

export function registerInputsCommands(program: Command) {
  const inputs = program.command('inputs').description('Lista/ajusta Inputs do indicador/EA ativo (CommandListener)');

  inputs
    .command('list')
    .option('--pattern <glob>', 'Filtro (contém)')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'LIST_INPUTS', [opts.pattern || ''], { timeoutMs: 8000 });
      if (resp.data.length === 0) {
        console.log(chalk.yellow('[inputs] nenhum input retornado (verifique se há indicador/EA ativo)'));
        return;
      }
      console.log(chalk.cyan('[inputs]'));
      resp.data.forEach((line) => console.log('  ' + line));
    });

  inputs
    .command('set')
    .requiredOption('--name <input>', 'Nome do input/variável')
    .requiredOption('--value <value>', 'Valor')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'SET_INPUT', [opts.name, opts.value], { timeoutMs: 6000 });
      console.log(chalk.green(`[inputs] set ${opts.name}=${opts.value} (${resp.message})`));
    });
}
