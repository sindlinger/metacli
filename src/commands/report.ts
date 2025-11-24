import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { normalizePath } from '../utils/paths.js';

const store = new ProjectStore();

function findLatest(dir: string, extensions: string[]): string | null {
  if (!fs.existsSync(dir)) return null;
  const files = fs.readdirSync(dir)
    .filter((f) => extensions.some((ext) => f.toLowerCase().endsWith(ext)))
    .map((f) => path.join(dir, f));
  if (files.length === 0) return null;
  files.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  return files[0];
}

export function registerReportCommands(program: Command) {
  const report = program.command('report').description('Acessa relatórios e screenshots');

  report
    .command('last')
    .option('--project <id>')
    .option('--copy <dir>', 'Copia para dir de saída')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const filesDir = path.join(info.data_dir, 'MQL5', 'Files');
      const reportsDir = path.join(filesDir, 'Tester');
      const latest = findLatest(reportsDir, ['.htm', '.html']);
      if (!latest) {
        console.log(chalk.yellow('[report] nenhum HTML encontrado em MQL5/Files/Tester'));
        return;
      }
      console.log(chalk.green(`[report] último: ${latest}`));
      if (opts.copy) {
        const outDir = normalizePath(opts.copy);
        await fs.ensureDir(outDir);
        const dest = path.join(outDir, path.basename(latest));
        await fs.copy(latest, dest, { overwrite: true });
        console.log(chalk.green(`[report] copiado para ${dest}`));
      }
    });

  report
    .command('shots')
    .description('Copia o screenshot mais recente (MQL5/Files/screenshots)')
    .option('--project <id>')
    .option('--copy <dir>', 'Diretório de destino')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const shotsDir = path.join(info.data_dir, 'MQL5', 'Files', 'screenshots');
      const latest = findLatest(shotsDir, ['.png', '.bmp', '.gif']);
      if (!latest) {
        console.log(chalk.yellow('[shots] nenhum screenshot encontrado'));
        return;
      }
      console.log(chalk.green(`[shots] último: ${latest}`));
      if (opts.copy) {
        const outDir = normalizePath(opts.copy);
        await fs.ensureDir(outDir);
        const dest = path.join(outDir, path.basename(latest));
        await fs.copy(latest, dest, { overwrite: true });
        console.log(chalk.green(`[shots] copiado para ${dest}`));
      }
    });
}
