import { Command } from 'commander';
import chalk from 'chalk';
import fs from 'fs-extra';
import path from 'path';
import { ProjectStore, ProjectInfo, repoRoot } from '../config/projectStore.js';
import { promptYesNo } from '../utils/prompt.js';
import { deployCommandListener } from '../utils/listenerDeploy.js';
import {
  deployFactoryTemplates,
  deployFactoryConfig,
  ensureAccountInIni,
  ensureCommandListenerStartup,
} from '../utils/factoryAssets.js';
import { logProjectSummary } from '../utils/projectSummary.js';
import { execa } from 'execa';
import { toWindowsPath } from '../utils/wsl.js';
import { resolvePowerShell } from '../utils/paths.js';

export function registerProjectCommands(program: Command) {
  const project = program.command('project').description('Gerencia apenas o registro de projetos (mtcli_projects.json)');

  const resolvePaths = (info: ProjectInfo): ProjectInfo => {
    const baseTerm = process.env.MTCLI_BASE_TERMINAL;
    const dataEnv = process.env.MTCLI_DATA_DIR;
    const terminalsDir = process.env.MTCLI_TERMINALS_DIR || path.join(repoRoot(), 'projects', 'terminals');
    const resolved: ProjectInfo = { ...info };
    if (!resolved.data_dir) {
      if (dataEnv) {
        resolved.data_dir = dataEnv;
      } else {
        // fallback padrão: projects/terminals/project-<id>
        resolved.data_dir = path.join(terminalsDir, `project-${info.project}`);
      }
    }
    if (!resolved.libs && resolved.data_dir) {
      resolved.libs = path.join(resolved.data_dir, 'MQL5', 'Libraries');
    }
    if (!resolved.terminal) {
      if (baseTerm) {
        resolved.terminal = path.join(baseTerm, 'terminal64.exe');
      } else if (resolved.data_dir) {
        // terminal e data_dir na mesma pasta (portable)
        resolved.terminal = path.join(resolved.data_dir, 'terminal64.exe');
      }
    }
    if (!resolved.metaeditor) {
      if (baseTerm) {
        resolved.metaeditor = path.join(baseTerm, 'metaeditor64.exe');
      } else if (resolved.data_dir) {
        resolved.metaeditor = path.join(resolved.data_dir, 'MetaEditor64.exe');
      }
    }
    // se nada informado, assume portable como padrão para cópia em projects/terminals
    if (!resolved.defaults) resolved.defaults = {};
    if (resolved.defaults.portable === undefined) {
      resolved.defaults.portable = true;
    }
    return resolved;
  };

  // kill desabilitado a pedido do usuário; mantém função no-code para não afetar a CLI
  const killTerminalProcesses = async (_info: ProjectInfo) => {
    console.log(chalk.yellow('[project] kill: desabilitado (não será executado).'));
  };

  const ensureProfile = async (dataDir: string, profileName: string) => {
    const chartsDir = path.join(dataDir, 'Profiles', 'Charts');
    const defaultDir = path.join(chartsDir, 'Default');
    const destDir = path.join(chartsDir, profileName);
    if (await fs.pathExists(destDir)) return;
    if (await fs.pathExists(defaultDir)) {
      await fs.copy(defaultDir, destDir);
      console.log(chalk.gray(`[project] profile: copiado Default -> ${destDir}`));
    } else {
      await fs.ensureDir(destDir);
      console.log(chalk.gray(`[project] profile: criado vazio ${destDir}`));
    }
    console.log(chalk.gray(`[project] profile "${profileName}" pronto`));
  };

  // start desabilitado a pedido do usuário
  const startTerminal = async (info: ProjectInfo, profileName: string) => {
    console.log(
      chalk.yellow(
        `[project] start: desabilitado. Para iniciar manualmente: start "" ${info.terminal} /portable /datapath:${info.data_dir} /profile:${profileName}`
      )
    );
  };

  // reload/remove reset desabilitados a pedido do usuário

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
