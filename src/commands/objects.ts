import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';
import fs from 'fs-extra';

const store = new ProjectStore();

export function registerObjectsCommands(program: Command) {
  return registerObjectsSubcommands(program.command('objects'));
}

export function registerObjectsSubcommands(parent: Command) {
  const objects = parent.command('objects').description('Opera objetos de gráfico (OBJ_*) via listener');

  objects
    .command('list')
    .option('-s, --symbol <symbol>')
    .option('-p, --period <period>')
    .option('--project <id>', '(LEGADO; evite, usa ativo)')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendListenerCommand(info, 'OBJ_LIST', [opts.symbol || '', opts.period || ''], { timeoutMs: 8000 });
    });

  objects
    .command('del')
    .description('Remove um objeto ou todos de um prefixo')
    .option('--name <name>', 'Nome do objeto')
    .option('--prefix <prefix>', 'Remove todos que iniciem com prefixo')
    .option('--project <id>', '(LEGADO; evite, usa ativo)')
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
    .option('--project <id>', '(LEGADO; evite, usa ativo)')
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

  objects
    .command('export')
    .description('Exporta objetos para JSON (depende de OBJ_LIST retornar dados)')
    .option('--project <id>')
    .option('--out <file>', 'Arquivo de saída', 'objects.json')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const resp = await sendListenerCommand(info, 'OBJ_LIST', ['', ''], { timeoutMs: 8000 });
      await fs.writeFile(opts.out, JSON.stringify(resp.data, null, 2), 'utf8');
      console.log(chalk.green(`[objects] exportado para ${opts.out}`));
    });

  objects
    .command('import')
    .description('Importa objetos de um JSON e cria via OBJ_CREATE')
    .requiredOption('--file <json>', 'Arquivo gerado pelo export')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const raw = await fs.readFile(opts.file, 'utf8');
      let entries: string[] = [];
      try {
        entries = JSON.parse(raw);
        if (!Array.isArray(entries)) throw new Error('JSON deve ser um array de strings');
      } catch (err) {
        throw new Error('Arquivo inválido para import (esperado array JSON de linhas do OBJ_LIST).');
      }
      for (const line of entries) {
        // espera formato "type|name|payloadJSON" devolvido pelo listener
        const parts = String(line).split('|');
        if (parts.length < 3) continue;
        const [type, name, payload] = parts;
        await sendListenerCommand(info, 'OBJ_CREATE', [type, name, payload], { timeoutMs: 8000 });
      }
      console.log(chalk.green(`[objects] import concluiu (${entries.length} itens)`));
    });
}
