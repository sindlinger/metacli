import { Command } from 'commander';

export function registerTesterCommands(program: Command) {
  const tester = program.command('tester').description('Opera o Strategy Tester');

  tester
    .command('status')
    .description('Mostra estado básico do tester')
    .action(() => {
      console.log('tester status: ainda não implementado neste CLI (placeholder).');
    });

  tester
    .command('run')
    .description('Placeholder para disparar testes automatizados')
    .action(() => {
      console.log('tester run: implemente o fluxo desejado aqui.');
    });
}
