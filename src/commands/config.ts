import { Command } from 'commander';
import chalk from 'chalk';
import { projectsFilePath } from '../config/projectStore.js';

export function registerConfigCommands(program: Command) {
  const config = program.command('config').description('Configurações globais do MTCLI');

  config
    .command('path')
    .description('Mostra o caminho do mtcli_projects.json')
    .action(() => {
      console.log(chalk.cyan(projectsFilePath()));
    });

  config
    .command('env')
    .description('Lista variáveis de ambiente relevantes para o mtcli')
    .action(() => {
      const envs = [
        'MTCLI_BASE_TERMINAL',
        'MTCLI_BASE_TERMINAL_DIR',
        'MTCLI_MT5_INSTALLER_URL',
        'MTCLI_PROJECTS',
        'MTCLI_DATA_DIR',
        'MTCLI_DEBUG',
      ];
      console.log(chalk.cyan('Variáveis de ambiente:'));
      envs.forEach((k) => console.log(`  ${k}=${process.env[k] ?? ''}`));
    });
}
