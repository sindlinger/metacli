import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, ProjectInfo } from '../config/projectStore.js';
import { normalizePath } from '../utils/paths.js';
import { restartListenerInstance } from './listener.js';

const store = new ProjectStore();

function resolveSymbol(info: ProjectInfo, fallback?: string) {
  return fallback || (info.defaults?.symbol as string | undefined);
}

function resolvePeriod(info: ProjectInfo, fallback?: string) {
  return fallback || (info.defaults?.period as string | undefined);
}

function resolveSubwindow(info: ProjectInfo, fallback?: number) {
  const value = fallback ?? (info.defaults?.subwindow as number | undefined);
  return typeof value === 'number' && !Number.isNaN(value) ? value : 1;
}

async function restartAndWrite(info: ProjectInfo, command: string) {
  await restartListenerInstance({ project: info.project, profile: info.defaults?.profile as string | undefined });
  await new Promise((resolve) => setTimeout(resolve, 1500));
  const dataDir = info.data_dir;
  if (!dataDir) {
    throw new Error('Projeto sem data_dir configurado.');
  }
  const fileDir = path.join(dataDir, 'MQL5', 'Files');
  await fs.ensureDir(fileDir);
  const filePath = path.join(fileDir, 'cmd.txt');
  await fs.writeFile(filePath, command, 'utf8');
  console.log(chalk.green(`Comando gravado em ${filePath}`));
}

async function copyTemplate(dataDir: string, templatePath: string, target: 'chart' | 'tester') {
  const src = normalizePath(templatePath);
  if (!(await fs.pathExists(src))) {
    throw new Error(`Template não encontrado: ${src}`);
  }
  const destDir = path.join(
    dataDir,
    'MQL5',
    'Profiles',
    target === 'tester' ? 'Tester' : 'Templates'
  );
  await fs.ensureDir(destDir);
  const dest = path.join(destDir, path.basename(src));
  await fs.copyFile(src, dest);
  console.log(chalk.green(`[template] ${src} -> ${dest}`));
}

export function registerChartCommands(program: Command) {
  const chart = program.command('chart').description('Opera gráficos e templates via listener');
  const indicator = chart.command('indicator').description('Gerencia indicadores via listener');

  indicator
    .command('attach')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .requiredOption('--indicator <name>', 'Nome do indicador (pastas relativas ao data_dir\\MQL5\\Indicators)')
    .option('--project <id>', 'Projeto configurado')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10))
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir. Configure via mtcli project save --data-dir ...');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const subwindow = resolveSubwindow(info, opts.subwindow);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `ATTACH_IND;${symbol};${period};${opts.indicator};${subwindow}`;
      await restartAndWrite(info, cmd);
    });

  indicator
    .command('detach')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `DETACH_IND;${symbol};${period}`;
      await restartAndWrite(info, cmd);
    });

  const template = chart.command('template').description('Gerencia templates (.tpl)');

  template
    .command('install')
    .requiredOption('--file <path>', 'Arquivo .tpl gerado no MT5')
    .option('--project <id>')
    .option('--target <chart|tester>', 'Onde instalar o template', 'chart')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const dataDir = info.data_dir;
      if (!dataDir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const target = opts.target === 'tester' ? 'tester' : 'chart';
      await copyTemplate(dataDir, opts.file, target);
    });

  template
    .command('apply')
    .requiredOption('--name <tpl>', 'Nome do template já instalado (ex.: WaveSpecZZ.tpl)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `APPLY_TPL;${symbol};${period};${opts.name}`;
      await restartAndWrite(info, cmd);
    });
}
