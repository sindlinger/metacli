import { Command } from 'commander';
import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import os from 'os';
import { ProjectStore } from '../config/projectStore.js';
import { resolveMetaeditorArgs } from '../utils/metaeditor.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';
import { toWslPath } from '../utils/paths.js';
import { execa } from 'execa';
import { resolveTarget } from '../utils/target.js';

const store = new ProjectStore();

function debounce(fn: () => Promise<void>, ms: number) {
  let timer: NodeJS.Timeout | null = null;
  return () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fn().catch((err) => console.error(err)), ms);
  };
}

export function registerWatchCommands(program: Command) {
  program
    .command('watch')
    .description('Vigia um .mq5: ao salvar, compila e reanexa no MT5')
    .option('--file <path>', 'Arquivo .mq5 a vigiar (Indicators/ ou Experts/)')
    .option('-i, --indicator <name>', 'Nome/atalho do indicador (pode ser I:ZigZag, ZigZag, ou subpasta)')
    .option('-e, --expert <name>', 'Nome/atalho do expert (pode ser E:MyEA, MyEA, ou subpasta)')
    .option('-s, --symbol <symbol>', 'Símbolo (default do projeto)')
    .option('-p, --period <period>', 'Período (default do projeto)')
    .option('--subwindow <index>', 'Subjanela do indicador', (val) => parseInt(val, 10))
    .option('--template <tpl>', 'Template para EA (default Default.tpl)')
    .option('--project <id>', '(LEGADO; evite, usa ativo)')
    .option('--debounce <ms>', 'Atraso entre detecção e recompilar', (v) => parseInt(v, 10), 400)
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.metaeditor) throw new Error('MetaEditor não configurado.');
      if (!info.data_dir) throw new Error('data_dir não configurado.');

      const target = resolveTarget(info, { file: opts.file, indicator: opts.indicator, expert: opts.expert });
      const isIndicator = target.kind === 'Indicators';
      const isExpert = target.kind === 'Experts';
      const attachName = target.attachName;
      const file = target.file;
      const symbol = opts.symbol || info.defaults?.symbol || 'EURUSD';
      const period = opts.period || info.defaults?.period || 'M1';
      const sub = typeof opts.subwindow === 'number' && Number.isFinite(opts.subwindow) ? opts.subwindow : info.defaults?.subwindow || 1;

      const compile = async () => {
        const args = resolveMetaeditorArgs(info, file);
        const metaExec = os.platform() === 'linux' ? toWslPath(info.metaeditor!) : info.metaeditor!;
        console.log(chalk.gray(`[watch] compilando ${file} ...`));
        const res = await execa(metaExec, args, { stdio: 'inherit', windowsHide: false, reject: false });
        if (res.exitCode !== 0) {
          console.log(chalk.yellow(`[watch] MetaEditor saiu com exitCode=${res.exitCode} (pode ser normal no WSL).`));
        }
      };

      const reattach = async () => {
        if (!symbol || !period) {
          console.log(chalk.yellow('[watch] pulei attach: defina symbol/period')); return;
        }
        if (isIndicator) {
          await sendListenerCommand(info, 'ATTACH_IND_FULL', [symbol, period, attachName, sub, ''], { timeoutMs: 8000 });
          console.log(chalk.green(`[watch] indicador reanexado: ${attachName} em ${symbol} ${period}`));
        } else {
          await sendListenerCommand(info, 'ATTACH_EA_FULL', [symbol, period, attachName, opts.template || 'Default.tpl', ''], { timeoutMs: 10000 });
          console.log(chalk.green(`[watch] expert reanexado: ${attachName} em ${symbol} ${period}`));
        }
      };

      const run = debounce(async () => {
        await compile();
        await reattach();
      }, opts.debounce);

      console.log(chalk.cyan(`[watch] observando ${file} ... Ctrl+C para sair`));
      await compile();
      await reattach();

      fs.watch(path.dirname(file), { persistent: true }, (event, fname) => {
        if (!fname) return;
        if (path.resolve(path.dirname(file), fname) === file && event === 'change') {
          run();
        }
      });
    });
}
