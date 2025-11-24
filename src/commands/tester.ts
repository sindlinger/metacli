import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, repoRoot } from '../config/projectStore.js';
import { normalizePath, toWinPath, toWslPath } from '../utils/paths.js';
import { runCommand } from '../utils/shell.js';

const store = new ProjectStore();
const DEFAULT_TESTER_INI = path.join(repoRoot(), 'tester_visual.ini');

function iniSet(filePath: string, section: string, key: string, value: string) {
  let lines: string[] = [];
  if (fs.existsSync(filePath)) {
    lines = fs.readFileSync(filePath, 'utf8').replace(/\r/g, '').split('\n');
  }
  let secStart = -1;
  let secEnd = lines.length;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('[') && line.endsWith(']')) {
      const name = line.slice(1, -1);
      if (name.toLowerCase() === section.toLowerCase()) {
        secStart = i;
      } else if (secStart >= 0) {
        secEnd = i;
        break;
      }
    }
  }
  if (secStart === -1) {
    lines.push(`[${section}]`);
    lines.push(`${key}=${value}`);
  } else {
    let found = false;
    for (let i = secStart + 1; i < secEnd; i++) {
      const line = lines[i];
      const idx = line.indexOf('=');
      if (idx > -1) {
        const k = line.slice(0, idx).trim();
        if (k.toLowerCase() === key.toLowerCase()) {
          lines[i] = `${key}=${value}`;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      lines.splice(secEnd, 0, `${key}=${value}`);
    }
  }
  fs.writeFileSync(filePath, lines.join('\n'), 'utf8');
}

export function registerTesterCommands(program: Command) {
  const tester = program.command('tester').description('Opera o Strategy Tester');

  tester
    .command('status')
    .description('Mostra estado básico do tester')
    .action(() => {
      console.log('tester status: consulte os relatórios/arquivos do MT5 (placeholder).');
    });

  tester
    .command('run')
    .description('Executa o terminal no modo Strategy Tester usando um arquivo .ini existente')
    .option('--config <path>', `Arquivo .ini (default: ${DEFAULT_TESTER_INI})`)
    .option('--project <id>', 'Projeto configurado (LEGADO; evite, usa ativo)')
    .option('--wait', 'Aguardar o terminal encerrar em vez de destacar', false)
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal) {
        throw new Error('terminal64.exe não configurado no projeto.');
      }
      if (!info.data_dir) {
        throw new Error('Defina data_dir no projeto (mtcli project save --data-dir ...).');
      }
      const iniPath = normalizePath(opts.config || DEFAULT_TESTER_INI);
      if (!(await fs.pathExists(iniPath))) {
        throw new Error(`Arquivo .ini não encontrado: ${iniPath}`);
      }
      const args = [`/config:${toWinPath(iniPath)}`, `/datapath:${toWinPath(info.data_dir)}`];
      const exe = toWslPath(info.terminal);
      const wait = Boolean(opts.wait);
      await runCommand(exe, args, { stdio: wait ? 'inherit' : 'ignore', detach: !wait });
      console.log(chalk.green(`Strategy Tester iniciado com ${iniPath}`));
    });

  tester
    .command('quick')
    .description('Gera ini rápido e roda tester (visual opcional)')
    .requiredOption('--expert <name>', 'Expert em Experts\\')
    .requiredOption('--symbol <symbol>', 'Símbolo')
    .requiredOption('--period <period>', 'Período (H1, M15, etc.)')
    .option('--from <YYYY.MM.DD>', 'Data inicial')
    .option('--to <YYYY.MM.DD>', 'Data final')
    .option('--visual', 'Modo visual', false)
    .option('--model <0|1|2>', 'Modelagem (0 every tick)', '0')
    .option('--spread <points>', 'Spread', '0')
    .option('--deposit <value>', 'Depósito', '10000')
    .option('--currency <ccy>', 'Moeda', 'USD')
    .option('--report <name>', 'Nome base do relatório', 'mtcli_run')
    .option('--project <id>', '(LEGADO; evite, usa ativo)')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal) throw new Error('terminal64.exe não configurado.');
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const iniDir = normalizePath(info.data_dir);
      const iniPath = path.join(iniDir, 'mtcli_quick.ini');
      const expertPath = opts.expert.toLowerCase().endsWith('.ex5') ? opts.expert : `${opts.expert}.ex5`;
      const lines = [
        '[Tester]',
        `Expert=Experts\\${expertPath}`,
        'ExpertParameters=',
        `Symbol=${opts.symbol}`,
        `Period=${opts.period}`,
        `Model=${opts.model}`,
        'Optimization=0',
        'ForwardMode=0',
        `Spread=${opts.spread}`,
        `Deposit=${opts.deposit}`,
        'Leverage=100',
        `Currency=${opts.currency}`,
        `Visual=${opts.visual ? 1 : 0}`,
        'ShutdownTerminal=0',
        `Report=${opts.report}`,
      ];
      if (opts.from) lines.push(`FromDate=${opts.from}`);
      if (opts.to) lines.push(`ToDate=${opts.to}`);
      lines.push('', '[TesterInputs]', '');
      await fs.writeFile(iniPath, lines.join('\n'), 'utf8');

      const exe = toWslPath(info.terminal);
      const args = [`/config:${toWinPath(iniPath)}`, `/datapath:${toWinPath(info.data_dir)}`];
      await runCommand(exe, args, { stdio: 'inherit', detach: false });
      console.log(chalk.green(`[tester] quick iniciado com ${iniPath}`));
    });

  tester
    .command('matrix')
    .description('Roda múltiplas combinações (símbolos/períodos) sequencialmente')
    .requiredOption('--expert <name>', 'Expert em Experts\\')
    .requiredOption('--symbols <list>', 'Lista separada por vírgulas')
    .requiredOption('--periods <list>', 'Lista separada por vírgulas')
    .option('--visual', 'Modo visual', false)
    .option('--from <YYYY.MM.DD>', 'Data inicial')
    .option('--to <YYYY.MM.DD>', 'Data final')
    .option('--project <id>')
    .action(async (opts) => {
      const symbols = String(opts.symbols).split(',').map((s: string) => s.trim()).filter(Boolean);
      const periods = String(opts.periods).split(',').map((s: string) => s.trim()).filter(Boolean);
      for (const symbol of symbols) {
        for (const period of periods) {
          const argsBase = ['--expert', opts.expert, '--symbol', symbol, '--period', period];
          if (opts.visual) argsBase.push('--visual');
          if (opts.from) argsBase.push('--from', opts.from);
          if (opts.to) argsBase.push('--to', opts.to);
          if (opts.project) argsBase.push('--project', opts.project);
          const { execa } = await import('execa');
          await execa('mtcli', ['tester', 'quick', ...argsBase], { stdio: 'inherit' });
        }
      }
    });

  tester
    .command('ini-show')
    .description('Mostra tester.ini do projeto (ou arquivo indicado)')
    .option('--file <path>', 'Arquivo ini (default tester.ini)')
    .option('--project <id>', 'Projeto alvo (LEGADO; evite, usa ativo)')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const iniPath = normalizePath(opts.file || path.join(info.data_dir!, 'tester.ini'));
      if (!(await fs.pathExists(iniPath))) {
        console.log(chalk.yellow(`tester ini não encontrado em ${iniPath}`));
        return;
      }
      const content = await fs.readFile(iniPath, 'utf8');
      console.log(content);
    });

  tester
    .command('ini-set')
    .description('Altera chave em tester.ini (default em data_dir/tester.ini)')
    .requiredOption('--key <key>')
    .requiredOption('--value <value>')
    .option('--section <name>', 'Seção (default Tester)', 'Tester')
    .option('--file <path>', 'Arquivo ini (default tester.ini)')
    .option('--project <id>', 'Projeto alvo (LEGADO; evite, usa ativo)')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const iniPath = normalizePath(opts.file || path.join(info.data_dir!, 'tester.ini'));
      await fs.ensureDir(path.dirname(iniPath));
      iniSet(iniPath, opts.section, opts.key, opts.value);
      console.log(chalk.green(`[tester] ${opts.section}.${opts.key}=${opts.value} em ${iniPath}`));
    });
}
