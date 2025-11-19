import { Command } from 'commander';

export function registerUtilsCommands(program: Command) {
  const utils = program.command('utils').description('Ferramentas auxiliares');

  utils
    .command('paths')
    .description('Mostra caminhos importantes do projeto')
    .action(() => {
      console.log('utils paths: conecte aqui os caminhos relevantes (placeholder).');
    });
}
