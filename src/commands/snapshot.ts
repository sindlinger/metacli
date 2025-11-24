import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

export function registerSnapshotCommands(program: Command) {
  const snap = program.command('snapshot').description('Salva/aplica snapshots de layout (charts/objs) via listener');

  snap
    .command('save')
    .requiredOption('--name <name>', 'Nome do snapshot')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'SNAPSHOT_SAVE', [opts.name], { timeoutMs: 10000 });
      console.log(chalk.green(`[snapshot] salvo: ${opts.name} (${resp.message})`));
    });

  snap
    .command('apply')
    .requiredOption('--name <name>', 'Nome do snapshot')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'SNAPSHOT_APPLY', [opts.name], { timeoutMs: 10000 });
      console.log(chalk.green(`[snapshot] aplicado: ${opts.name} (${resp.message})`));
    });

  snap
    .command('list')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'SNAPSHOT_LIST', [], { timeoutMs: 8000 });
      if (resp.data.length === 0) {
        console.log(chalk.yellow('[snapshot] nenhum snapshot listado'));
        return;
      }
      console.log(chalk.cyan('[snapshot]'));
      resp.data.forEach((line) => console.log('  ' + line));
    });
}
