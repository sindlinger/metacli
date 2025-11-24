import { Command } from 'commander';

export function registerEditorCommands(program: Command) {
  const editor = program.command('editor').description('Integra com o MetaEditor');

  editor
    .command('compile')
    .description('Compila um arquivo .mq5/.mqh usando o MetaEditor configurado no projeto')
    .requiredOption('--file <path>', 'Arquivo MQL a compilar (mq5/mqh)')
    .option('--project <id>', 'Projeto configurado')
    .option('--log <path>', 'Caminho do log gerado pelo MetaEditor')
    .action(async (opts) => {
      const { ProjectStore } = await import('../config/projectStore.js');
      const { resolveMetaeditorArgs } = await import('../utils/metaeditor.js');
      const { execa } = await import('execa');
      const store = new ProjectStore();
      const info = await store.useOrThrow(opts.project);
      const metaeditor = info.metaeditor;
      if (!metaeditor) throw new Error('Defina metaeditor no projeto (ex.: C:/.../MetaEditor64.exe)');
      const args = resolveMetaeditorArgs(info, opts.file, opts.log);
      await execa(metaeditor, args, { stdio: 'inherit', windowsHide: false });
    });
}
