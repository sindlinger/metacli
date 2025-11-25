import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { promptYesNo } from '../utils/prompt.js';
import { deployCommandListener } from '../utils/listenerDeploy.js';
import { deployFactoryTemplates, deployFactoryConfig, ensureAccountInIni } from '../utils/factoryAssets.js';
import { restartListenerInstance } from './listener.js';
import { logProjectSummary } from '../utils/projectSummary.js';

export function registerProjectCommands(program: Command) {
  const project = program.command('project').description('Gerencia apenas o registro de projetos (mtcli_projects.json)');

  project
    .command('reset')
    .description('Reaplica listener/config/templates no projeto atual e reinicia o listener (sem reinstalar terminal)')
    .option('--id <id>', 'Projeto alvo (default: projeto ativo/last_project)')
    .action(async (opts) => {
      const store = new ProjectStore();
      const current = await store.useOrThrow(opts.id);
      const file = await store.show();
      const info = file.projects[current.project];
      if (!info) {
        throw new Error(`Projeto "${current.project}" não encontrado em mtcli_projects.json.`);
      }

      if (!info.terminal || !info.metaeditor || !info.data_dir || !info.libs) {
        throw new Error('Projeto não possui caminhos completos (terminal/metaeditor/data_dir/libs).');
      }

      const install = {
        terminal: info.terminal,
        metaeditor: info.metaeditor,
        dataDir: info.data_dir,
        libs: info.libs,
        root: '',
      };

      const saved = await store.setProject(current.project, info, true);

      await deployCommandListener(install);
      await deployFactoryTemplates(install.dataDir);
      await deployFactoryConfig(install.dataDir);
      await ensureAccountInIni(install.dataDir);

      await restartListenerInstance({ project: saved.project, profile: saved.defaults?.profile ?? undefined });
      await logProjectSummary(saved);
      console.log(chalk.green(`Projeto ${saved.project} resetado.`));
    });

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
