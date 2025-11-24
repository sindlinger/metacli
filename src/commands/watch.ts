import { Command } from 'commander';
import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { resolveMetaeditorArgs } from '../utils/metaeditor.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';
import { normalizePath } from '../utils/paths.js';
import { execa } from 'execa';

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
    .requiredOption('--file <path>', 'Arquivo .mq5 a vigiar (Indicators/ ou Experts/)')
    .option('-i, --indicator <name>', 'Nome para anexar (se Indicators)')
    .option('-e, --expert <name>', 'Nome para anexar (se Experts)')
    .option('-s, --symbol <symbol>', 'Símbolo (default do projeto)')
    .option('-p, --period <period>', 'Período (default do projeto)')
    .option('--subwindow <index>', 'Subjanela do indicador', (val) => parseInt(val, 10))
    .option('--template <tpl>', 'Template para EA (default Default.tpl)')
    .option('--project <id>')
    .option('--debounce <ms>', 'Atraso entre detecção e recompilar', (v) => parseInt(v, 10), 400)
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.metaeditor) throw new Error('MetaEditor não configurado.');
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const file = normalizePath(opts.file);
      if (!fs.existsSync(file)) throw new Error(`Arquivo não encontrado: ${file}`);
      const isIndicator = Boolean(opts.indicator) || file.toLowerCase().includes('indicator');
      const isExpert = Boolean(opts.expert) || file.toLowerCase().includes('expert');
      if (isIndicator === isExpert) {
        throw new Error('Informe --indicator ou --expert para saber como anexar.');
      }
      const symbol = opts.symbol || info.defaults?.symbol;
      const period = opts.period || info.defaults?.period;
      const sub = typeof opts.subwindow === 'number' && Number.isFinite(opts.subwindow) ? opts.subwindow : info.defaults?.subwindow || 1;

      const compile = async () => {
        const args = resolveMetaeditorArgs(info, file);
        console.log(chalk.gray(`[watch] compilando ${file} ...`));
        await execa(info.metaeditor!, args, { stdio: 'inherit', windowsHide: false });
      };

      const reattach = async () => {
        if (!symbol || !period) {
          console.log(chalk.yellow('[watch] pulei attach: defina symbol/period')); return;
        }
        if (isIndicator) {
          await sendListenerCommand(info, 'ATTACH_IND_FULL', [symbol, period, opts.indicator, sub, ''], { timeoutMs: 8000 });
          console.log(chalk.green(`[watch] indicador reanexado: ${opts.indicator} em ${symbol} ${period}`));
        } else {
          await sendListenerCommand(info, 'ATTACH_EA_FULL', [symbol, period, opts.expert, opts.template || 'Default.tpl', ''], { timeoutMs: 10000 });
          console.log(chalk.green(`[watch] expert reanexado: ${opts.expert} em ${symbol} ${period}`));
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
