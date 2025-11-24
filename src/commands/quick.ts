import { Command } from 'commander';
import path from 'path';
import chalk from 'chalk';
import fs from 'fs-extra';
import { ProjectStore } from '../config/projectStore.js';
import { resolveMetaeditorArgs } from '../utils/metaeditor.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';
import { execa } from 'execa';
import { normalizePath, toWinPath, toWslPath } from '../utils/paths.js';
import { runCommand } from '../utils/shell.js';

const store = new ProjectStore();

function resolveSymbol(info: any, fallback?: string) {
  return fallback || (info.defaults?.symbol as string | undefined);
}

function resolvePeriod(info: any, fallback?: string) {
  return fallback || (info.defaults?.period as string | undefined);
}

function resolveSubwindow(info: any, fallback?: number) {
  const value = fallback ?? (info.defaults?.subwindow as number | undefined);
  return typeof value === 'number' && !Number.isNaN(value) ? value : 1;
}

function normalizeMqlPath(kind: 'indicator' | 'expert' | 'script', input: string, dataDir: string): string {
  const trimmed = input.replace(/^\\+|^\/+/g, '');
  const expectedRoot = kind === 'indicator' ? 'indicators' : kind === 'expert' ? 'experts' : 'scripts';
  const parts = trimmed.split(/\\|\//);
  const root = (parts[0] || '').toLowerCase();
  if (root !== expectedRoot) {
    throw new Error(`Use um caminho que comece com \\${expectedRoot}\\ (ex.: \\Indicators\\MeuIndi.mq5).`);
  }
  let file = parts.join(path.sep);
  if (!path.extname(file)) {
    file = `${file}.mq5`;
  }
  return path.join(dataDir, 'MQL5', file);
}

async function compileWithMetaeditor(info: any, targetFile: string) {
  if (!info.metaeditor) throw new Error('MetaEditor não configurado para o projeto.');
  const args = resolveMetaeditorArgs(info, targetFile);
  await execa(info.metaeditor, args, { stdio: 'inherit', windowsHide: false });
}

async function attachIndicator(info: any, symbol: string, period: string, indicator: string, subwindow: number) {
  await sendListenerCommand(info, 'ATTACH_IND_FULL', [symbol, period, indicator, subwindow], { timeoutMs: 8000 });
}

async function detachIndicator(info: any, symbol: string, period: string, indicator: string, subwindow: number) {
  await sendListenerCommand(info, 'DETACH_IND_FULL', [symbol, period, indicator, subwindow], { timeoutMs: 8000 });
}

async function attachExpert(info: any, symbol: string, period: string, expert: string, template = 'Default.tpl') {
  await sendListenerCommand(info, 'ATTACH_EA_FULL', [symbol, period, expert, template], { timeoutMs: 10000 });
}

async function detachExpert(info: any, symbol: string, period: string) {
  await sendListenerCommand(info, 'DETACH_EA_FULL', [symbol, period], { timeoutMs: 8000 });
}

async function indicatorTotal(info: any, symbol: string, period: string, subwindow: number) {
  await sendListenerCommand(info, 'IND_TOTAL', [symbol, period, subwindow], { timeoutMs: 6000 });
}

async function indicatorName(info: any, symbol: string, period: string, subwindow: number, index: number) {
  await sendListenerCommand(info, 'IND_NAME', [symbol, period, subwindow, index], { timeoutMs: 6000 });
}

async function indicatorHandle(info: any, symbol: string, period: string, subwindow: number, name: string) {
  await sendListenerCommand(info, 'IND_HANDLE', [symbol, period, subwindow, name], { timeoutMs: 6000 });
}

async function runExpertVisual(info: any, expert: string, symbol: string, period: string, configPath?: string, wait = false) {
  if (!info.terminal) throw new Error('terminal64.exe não configurado para o projeto.');
  if (!info.data_dir) throw new Error('data_dir não configurado para o projeto.');

  const expertPath = expert.toLowerCase().endsWith('.ex5') ? expert : `${expert}.ex5`;
  let iniPath = configPath ? normalizePath(configPath) : undefined;
  if (!iniPath) {
    const iniDir = path.dirname(info.data_dir);
    await fs.ensureDir(iniDir);
    iniPath = path.join(iniDir, 'run_visual.ini');
    const content = [
      '[Tester]',
      `Expert=Experts\\${expertPath}`,
      'ExpertParameters=',
      `Symbol=${symbol}`,
      `Period=${period}`,
      'Model=0',
      'Optimization=0',
      'ForwardMode=0',
      'Spread=0',
      'Deposit=10000',
      'Leverage=100',
      'Currency=USD',
      'Visual=1',
      'ShutdownTerminal=0',
      'Report=run_report',
      '',
      '[TesterInputs]',
      '',
    ].join('\n');
    await fs.writeFile(iniPath, content, 'utf8');
  }

  const exe = toWslPath(info.terminal);
  const args = [`/config:${toWinPath(iniPath)}`, `/datapath:${toWinPath(info.data_dir)}`];
  await runCommand(exe, args, { stdio: 'inherit', detach: !wait });
}

async function listMqlFiles(info: any, kind: 'indicator' | 'expert' | 'script') {
  if (!info.data_dir) throw new Error('data_dir não configurado para o projeto.');
  const root = path.join(info.data_dir, 'MQL5', kind === 'indicator' ? 'Indicators' : kind === 'expert' ? 'Experts' : 'Scripts');
  if (!(await fs.pathExists(root))) {
    console.log(chalk.yellow(`Pasta não encontrada: ${root}`));
    return;
  }
  const files: string[] = [];
  const walk = async (dir: string) => {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (/\.(mq5|ex5)$/i.test(entry.name)) {
        files.push(path.relative(root, full));
      }
    }
  };
  await walk(root);
  if (files.length === 0) {
    console.log(chalk.yellow('Nenhum arquivo encontrado.'));
    return;
  }
  console.log(chalk.cyan(`Lista (${kind}):`));
  files.sort().forEach((f) => console.log('  ' + f));
}

async function listAttached(info: any) {
  const result = await sendListenerCommand(info, 'LIST_CHARTS', [], { timeoutMs: 8000 });
  if (result.data.length > 0) {
    console.log(chalk.cyan('Charts ativos:'));
    for (const line of result.data) {
      console.log('  ' + line);
    }
  } else {
    console.log(chalk.yellow('Nenhum chart listado pelo listener.'));
  }
}

export function registerQuickCommands(program: Command) {
  program
    .command('compile')
    .description('Compila rapidamente um indicador/expert/script (caminho começa em \\Indicators\\, \\Experts\\ ou \\Scripts\\)')
    .option('-i, --indicator <path>', 'Caminho sob MQL5/Indicators')
    .option('-e, --expert <path>', 'Caminho sob MQL5/Experts')
    .option('-s, --script <path>', 'Caminho sob MQL5/Scripts')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const kinds = ['indicator', 'expert', 'script'].filter((k) => (opts as any)[k]);
      if (kinds.length !== 1) {
        throw new Error('Informe exatamente um alvo: --indicator ou --expert ou --script.');
      }
      const kind = kinds[0] as 'indicator' | 'expert' | 'script';
      const raw = (opts as any)[kind] as string;
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado para o projeto.');
      const targetFile = normalizeMqlPath(kind, raw, normalizePath(info.data_dir));
      if (!(await fs.pathExists(targetFile))) {
        throw new Error(`Arquivo não encontrado: ${targetFile}`);
      }
      await compileWithMetaeditor(info, targetFile);
      console.log(chalk.green(`[compile] OK: ${targetFile}`));
    });

  program
    .command('run')
    .description('Roda indicador/expert no chart ou tester (visual com -v)')
    .option('-i, --indicator <name>', 'Nome do indicador (ex.: Examples\\ZigZag)')
    .option('-e, --expert <name>', 'Nome do expert (ex.: Moving_Average)')
    .option('-t, --template <tpl>', 'Template para anexar o expert (default: Default.tpl)')
    .option('-v, --visual', 'Usa Strategy Tester em modo visual (apenas expert)', false)
    .option('-s, --symbol <symbol>', 'Símbolo (default do projeto)')
    .option('-p, --period <period>', 'Período (default do projeto)')
    .option('--subwindow <index>', 'Subjanela do indicador', (val) => parseInt(val, 10))
    .option('--config <path>', 'Arquivo .ini customizado para tester (expert + --visual)')
    .option('--wait', 'Aguardar o terminal fechar (tester visual/config)', false)
    .option('--remove', 'Remove (detach) em vez de anexar')
    .option('--redraw', 'Detach + attach do indicador')
    .option('--total', 'Mostra ChartIndicatorsTotal (indicator)')
    .option('--name', 'Mostra ChartIndicatorName (indicator)')
    .option('--handle', 'Mostra handle ChartIndicatorGet (indicator)')
    .option('--index <n>', 'Índice para --name (default 0)', (val) => parseInt(val, 10))
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const targets = ['indicator', 'expert'].filter((k) => (opts as any)[k]);
      if (targets.length !== 1) {
        throw new Error('Use --indicator OU --expert.');
      }
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }

      if (opts.indicator) {
        const sub = resolveSubwindow(info, opts.subwindow);
        const idx = typeof opts.index === 'number' && Number.isFinite(opts.index) ? opts.index : 0;

        if (opts.total) {
          await indicatorTotal(info, symbol, period, sub);
          return;
        }
        if (opts.name) {
          await indicatorName(info, symbol, period, sub, idx);
          return;
        }
        if (opts.handle) {
          await indicatorHandle(info, symbol, period, sub, opts.indicator);
          return;
        }
        if (opts.remove) {
          await detachIndicator(info, symbol, period, opts.indicator, sub);
          console.log(chalk.green(`[run] Indicador removido: ${opts.indicator} de ${symbol} ${period} (subwindow ${sub})`));
          return;
        }
        if (opts.redraw) {
          await detachIndicator(info, symbol, period, opts.indicator, sub);
          await attachIndicator(info, symbol, period, opts.indicator, sub);
          console.log(chalk.green(`[run] Indicador redesenhado: ${opts.indicator} em ${symbol} ${period} (subwindow ${sub})`));
          return;
        }

        await attachIndicator(info, symbol, period, opts.indicator, sub);
        console.log(chalk.green(`[run] Indicador anexado: ${opts.indicator} em ${symbol} ${period} (subwindow ${sub})`));
        return;
      }

      // expert
      if (opts.remove) {
        await detachExpert(info, symbol, period);
        console.log(chalk.green(`[run] Expert removido de ${symbol} ${period}`));
        return;
      }

      if (opts.visual) {
        await runExpertVisual(info, opts.expert, symbol, period, opts.config, opts.wait);
        console.log(chalk.green(`[run] Tester visual iniciado com expert ${opts.expert} em ${symbol} ${period}`));
      } else {
        await attachExpert(info, symbol, period, opts.expert, opts.template);
        console.log(chalk.green(`[run] Expert anexado: ${opts.expert} em ${symbol} ${period}`));
      }
    });

  program
    .command('list')
    .description('Lista indicadores/experts/scripts no MQL5 do projeto')
    .option('-i, --indicator', 'Listar Indicators')
    .option('-e, --expert', 'Listar Experts')
    .option('-s, --script', 'Listar Scripts')
    .option('-a, --attached', 'Listar charts/indicadores anexados (listener)')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if ((opts as any).attached) {
        await listAttached(info);
        return;
      }
      const flags = ['indicator', 'expert', 'script'].filter((k) => (opts as any)[k]);
      const selected = flags.length === 0 ? ['indicator'] : flags;
      for (const kind of selected as ('indicator' | 'expert' | 'script')[]) {
        await listMqlFiles(info, kind);
      }
    });
}
