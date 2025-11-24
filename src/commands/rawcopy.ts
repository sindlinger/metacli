import { Command } from 'commander';
import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { normalizePath } from '../utils/paths.js';

const store = new ProjectStore();

export function registerCopyCommands(program: Command) {
  program
    .command('copy')
    .description('Copia arquivos arbitrários para dentro do data_dir/MQL5 (útil para dll/ex5)')
    .requiredOption('--src <file>', 'Arquivo a copiar')
    .requiredOption('--dest <path>', 'Destino relativo ao MQL5 (ex.: Libraries/xxx.dll)')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const src = normalizePath(opts.src);
      if (!(await fs.pathExists(src))) throw new Error(`Fonte não encontrada: ${src}`);
      const dest = normalizePath(path.join(info.data_dir, 'MQL5', opts.dest));
      await fs.ensureDir(path.dirname(dest));
      await fs.copy(src, dest, { overwrite: true });
      console.log(chalk.green(`[copy] ${src} -> ${dest}`));
    });
}
