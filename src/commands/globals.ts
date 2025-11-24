import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

function requireOne<T>(values: Array<[string, T | undefined]>): [string, T] {
  const present = values.filter(([, v]) => v !== undefined && v !== null && v !== '');
  if (present.length !== 1) {
    throw new Error('Informe exatamente um alvo (ex.: --name ou --prefix).');
  }
  const [label, value] = present[0];
  return [label, value as T];
}

export function registerGlobalsCommands(program: Command) {
  const globals = program.command('globals').description('Gerencia Global Variables do MT5 via listener');

  globals
    .command('set')
    .requiredOption('--name <name>', 'Nome da Global Variable')
    .requiredOption('--value <number>', 'Valor numérico (double)')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const value = Number(opts.value);
      if (!Number.isFinite(value)) {
        throw new Error('Informe --value numérico.');
      }
      const resp = await sendListenerCommand(info, 'GLOBAL_SET', [opts.name, value], { timeoutMs: 6000 });
      console.log(chalk.green(`[globals] set ${opts.name} = ${value} (${resp.message})`));
    });

  globals
    .command('get')
    .requiredOption('--name <name>', 'Nome da Global Variable')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'GLOBAL_GET', [opts.name], { timeoutMs: 6000 });
      const value = resp.data[0] ?? '(sem valor)';
      console.log(chalk.green(`[globals] ${opts.name} = ${value}`));
    });

  globals
    .command('del')
    .description('Remove Global Variables')
    .option('--name <name>', 'Remove uma variável específica')
    .option('--prefix <prefix>', 'Remove todas que iniciem com o prefixo')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const [kind, target] = requireOne<string>([
        ['name', opts.name],
        ['prefix', opts.prefix],
      ]);
      const type = kind === 'name' ? 'GLOBAL_DEL' : 'GLOBAL_DEL_PREFIX';
      const resp = await sendListenerCommand(info, type, [target], { timeoutMs: 6000 });
      console.log(chalk.green(`[globals] removido (${kind}=${target}): ${resp.message}`));
    });

  globals
    .command('list')
    .description('Lista Global Variables')
    .option('--prefix <prefix>', 'Filtra por prefixo')
    .option('--limit <n>', 'Limite de resultados', (val) => parseInt(val, 10))
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const limit = Number.isFinite(opts.limit) ? opts.limit : undefined;
      const resp = await sendListenerCommand(info, 'GLOBAL_LIST', [opts.prefix, limit ?? ''], { timeoutMs: 8000 });
      if (resp.data.length === 0) {
        console.log(chalk.yellow('[globals] nenhuma variável encontrada'));
        return;
      }
      console.log(chalk.cyan('[globals]')); 
      for (const line of resp.data) {
        console.log('  ' + line);
      }
    });
}
