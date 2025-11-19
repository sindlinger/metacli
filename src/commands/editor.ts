import { Command } from 'commander';

export function registerEditorCommands(program: Command) {
  const editor = program.command('editor').description('Integra com o MetaEditor');

  editor
    .command('compile')
    .description('Compila um arquivo MQL (placeholder)')
    .option('--file <path>')
    .action((opts) => {
      if (!opts.file) {
        console.log('editor compile: especifique --file para compilar (em desenvolvimento).');
        return;
      }
      console.log(`editor compile: fluxo para ${opts.file} ainda não implementado.`);
    });
}
