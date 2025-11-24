import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

export function registerTradeCommands(program: Command) {
  const trade = program.command('trade').description('Ordens simples via listener (requer handlers TRADE_*)');

  trade
    .command('buy')
    .requiredOption('--symbol <symbol>')
    .requiredOption('--volume <lot>', 'Volume em lotes')
    .option('--sl <price>', 'Stop loss')
    .option('--tp <price>', 'Take profit')
    .option('--comment <text>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'TRADE_BUY', [opts.symbol, opts.volume, opts.sl || '', opts.tp || '', opts.comment || ''], { timeoutMs: 8000 });
      console.log(chalk.green(`[trade] buy: ${resp.message}`));
    });

  trade
    .command('sell')
    .requiredOption('--symbol <symbol>')
    .requiredOption('--volume <lot>', 'Volume em lotes')
    .option('--sl <price>', 'Stop loss')
    .option('--tp <price>', 'Take profit')
    .option('--comment <text>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'TRADE_SELL', [opts.symbol, opts.volume, opts.sl || '', opts.tp || '', opts.comment || ''], { timeoutMs: 8000 });
      console.log(chalk.green(`[trade] sell: ${resp.message}`));
    });

  trade
    .command('close-all')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'TRADE_CLOSE_ALL', [], { timeoutMs: 10000 });
      console.log(chalk.green(`[trade] close-all: ${resp.message}`));
    });

  trade
    .command('list')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'TRADE_LIST', [], { timeoutMs: 8000 });
      if (resp.data.length === 0) {
        console.log(chalk.yellow('[trade] nenhuma ordem retornada'));
        return;
      }
      resp.data.forEach((line) => console.log(line));
    });
}
