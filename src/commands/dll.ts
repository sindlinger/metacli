import { Command } from 'commander';

export function registerDllCommands(program: Command) {
  const dll = program.command('dll').description('Builds e utilitários das DLLs');

  dll
    .command('build')
    .description('Fluxo de build das DLLs (placeholder)')
    .action(() => {
      console.log('dll build: conecte com seu pipeline de build (em desenvolvimento).');
    });
}
