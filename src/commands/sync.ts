import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { normalizePath } from '../utils/paths.js';

const store = new ProjectStore();

const TARGET_MAP: Record<string, string> = {
  indicators: path.join('MQL5', 'Indicators'),
  experts: path.join('MQL5', 'Experts'),
  scripts: path.join('MQL5', 'Scripts'),
  libraries: path.join('MQL5', 'Libraries'),
};

export function registerSyncCommands(program: Command) {
  program
    .command('sync')
    .description('Copia artefatos (ex5/dll) para o data_dir do projeto')
    .requiredOption('--from <dir>', 'Diretório de origem (build/output)')
    .requiredOption('--target <indicators|experts|scripts|libraries>', 'Destino dentro do MQL5')
    .option('--pattern <glob>', 'Filtro simples por extensão (ex.: .ex5, .dll)', '.ex5')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const rel = TARGET_MAP[opts.target];
      if (!rel) throw new Error('target deve ser indicators|experts|scripts|libraries');
      const srcDir = normalizePath(opts.from);
      const dstDir = normalizePath(path.join(info.data_dir, rel));
      if (!(await fs.pathExists(srcDir))) throw new Error(`Origem não encontrada: ${srcDir}`);
      await fs.ensureDir(dstDir);
      const pattern = (opts.pattern as string) || '.ex5';
      const files = (await fs.readdir(srcDir)).filter((f) => f.toLowerCase().endsWith(pattern.toLowerCase()));
      if (files.length === 0) {
        console.log(chalk.yellow(`[sync] nenhum arquivo ${pattern} em ${srcDir}`));
        return;
      }
      for (const f of files) {
        await fs.copy(path.join(srcDir, f), path.join(dstDir, f), { overwrite: true });
        console.log(chalk.gray(`[sync] ${f} -> ${dstDir}`));
      }
      console.log(chalk.green(`[sync] ${files.length} arquivo(s) copiado(s)`));
    });
}
