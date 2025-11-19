import { Command } from 'commander';
import { projectsFilePath } from '../config/projectStore.js';

export function registerConfigCommands(program: Command) {
  const config = program.command('config').description('Configurações globais do MTCLI');

  config
    .command('path')
    .description('Mostra o caminho do mtcli_projects.json')
    .action(() => {
      console.log(projectsFilePath());
    });
}
