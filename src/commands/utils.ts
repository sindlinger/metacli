import { Command } from 'commander';
import chalk from 'chalk';
import path from 'path';
import { ProjectStore } from '../config/projectStore.js';
import { projectsFilePath } from '../config/projectStore.js';

export function registerUtilsCommands(program: Command) {
  const utils = program.command('utils').description('Ferramentas auxiliares');

  utils
    .command('paths')
    .description('Mostra caminhos importantes do projeto/data_dir')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const store = new ProjectStore();
      const info = await store.useOrThrow(opts.project);
      const rows: Array<[string, string | undefined]> = [
        ['projects_file', projectsFilePath()],
        ['project', info.project],
        ['data_dir', info.data_dir],
        ['terminal', info.terminal],
        ['metaeditor', info.metaeditor],
        ['libs', info.libs],
      ];
      console.log(chalk.cyan('utils paths:'));
      rows.forEach(([k, v]) => console.log(`  ${k}: ${v ?? '(unset)'}`));
    });
}
