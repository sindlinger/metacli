import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';

const store = new ProjectStore();

async function writeCommandFile(dataDir: string, payload: string) {
  const fileDir = path.join(dataDir, 'MQL5', 'Files');
  await fs.ensureDir(fileDir);
  const filePath = path.join(fileDir, 'cmd.txt');
  await fs.writeFile(filePath, payload, 'utf8');
  console.log(chalk.green(`Comando gravado em ${filePath}`));
}

export function registerChartCommands(program: Command) {
  const chart = program.command('chart').description('Opera gráficos através do listener');
  const indicator = chart.command('indicator').description('Gerencia indicadores via listener');

  indicator
    .command('attach')
    .requiredOption('--symbol <symbol>')
    .requiredOption('--period <period>')
    .requiredOption('--indicator <name>', 'Nome do indicador (pastas relativas ao data_dir\\MQL5\\Indicators)')
    .option('--project <id>', 'Projeto configurado')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10), 1)
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const dataDir = info.data_dir;
      if (!dataDir) {
        throw new Error('Projeto sem data_dir. Configure via mtcli project save --data-dir ...');
      }
      const cmd = `ATTACH_IND;${opts.symbol};${opts.period};${opts.indicator};${opts.subwindow}`;
      await writeCommandFile(dataDir, cmd);
    });

  indicator
    .command('detach')
    .requiredOption('--symbol <symbol>')
    .requiredOption('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const dataDir = info.data_dir;
      if (!dataDir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const cmd = `DETACH_IND;${opts.symbol};${opts.period}`;
      await writeCommandFile(dataDir, cmd);
    });
}
