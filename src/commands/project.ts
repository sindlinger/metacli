import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';

export function registerProjectCommands(program: Command) {
  const project = program.command('project').description('Gerencia apenas o registro de projetos (mtcli_projects.json)');

  project
    .command('remove')
    .alias('rm')
    .description('Remove um projeto do mtcli (não apaga arquivos, só o registro)')
    .option('--id <id>', 'Projeto a remover (default: projeto ativo/last_project)')
    .action(async (opts) => {
      const store = new ProjectStore();
      const current = await store.useOrThrow(opts.id);
      const file = await store.show();
      const target = current.project;
      if (!file.projects[target]) {
        throw new Error(`Projeto "${target}" não encontrado em mtcli_projects.json.`);
      }

      delete file.projects[target];

      if (file.last_project === target) {
        const remaining = Object.keys(file.projects);
        file.last_project = remaining.length > 0 ? remaining[0] : undefined;
      }

      await store.save(file);
      console.log(chalk.green(`Projeto "${target}" removido do registro. (Arquivos em projects/terminals não foram apagados.)`));
      if (file.last_project) {
        console.log(chalk.gray(`last_project agora: ${file.last_project}`));
      }
    });
}
