import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

export function registerObjectsCommands(program: Command) {
  const objects = program.command('objects').description('Opera objetos de gráfico (OBJ_*) via listener');

  objects
    .command('list')
    .option('-s, --symbol <symbol>')
    .option('-p, --period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendListenerCommand(info, 'OBJ_LIST', [opts.symbol || '', opts.period || ''], { timeoutMs: 8000 });
    });

  objects
    .command('del')
    .description('Remove um objeto ou todos de um prefixo')
    .option('--name <name>', 'Nome do objeto')
    .option('--prefix <prefix>', 'Remove todos que iniciem com prefixo')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!opts.name && !opts.prefix) {
        throw new Error('Informe --name ou --prefix');
      }
      const type = opts.prefix ? 'OBJ_DELETE_PREFIX' : 'OBJ_DELETE';
      const value = opts.prefix || opts.name;
      await sendListenerCommand(info, type, [value], { timeoutMs: 6000 });
      console.log(chalk.green(`[objects] removido (${type} ${value})`));
    });

  objects
    .command('move')
    .requiredOption('--name <name>')
    .requiredOption('--price <price>', 'Preço (y)')
    .requiredOption('--time <time>', 'Tempo (x) YYYY.MM.DD HH:MM')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendListenerCommand(info, 'OBJ_MOVE', [opts.name, opts.price, opts.time], { timeoutMs: 6000 });
      console.log(chalk.green(`[objects] move ${opts.name} -> ${opts.time} @ ${opts.price}`));
    });

  objects
    .command('create')
    .requiredOption('--type <OBJ_TYPE>', 'Ex.: OBJ_TREND, OBJ_HLINE, OBJ_TEXT')
    .requiredOption('--name <name>')
    .requiredOption('--time <time>', 'Tempo inicial')
    .requiredOption('--price <price>', 'Preço inicial')
    .option('--time2 <time>', 'Tempo final (linhas)')
    .option('--price2 <price>', 'Preço final (linhas)')
    .option('--text <text>', 'Texto (OBJ_TEXT)')
    .option('--color <color>', 'Cor (ex.: clrRed)')
    .option('--style <style>', 'Estilo (linha)')
    .option('--width <n>', 'Espessura')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const payload = JSON.stringify({
        time: opts.time,
        price: opts.price,
        time2: opts.time2,
        price2: opts.price2,
        text: opts.text,
        color: opts.color,
        style: opts.style,
        width: opts.width,
      });
      await sendListenerCommand(info, 'OBJ_CREATE', [opts.type, opts.name, payload], { timeoutMs: 8000 });
      console.log(chalk.green(`[objects] create ${opts.type} ${opts.name}`));
    });
}
