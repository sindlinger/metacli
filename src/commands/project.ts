import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore, ProjectDefaults } from '../config/projectStore.js';
import { restartListenerInstance } from './listener.js';
import { DEFAULT_AGENT_DEFAULTS, DEFAULT_AGENT_PROJECT_ID } from '../config/agentDefaults.js';
import { logProjectSummary } from '../utils/projectSummary.js';
import { promptYesNo } from '../utils/prompt.js';
import { ensureBaseTerminal, installTerminalForProject } from '../utils/terminalInstall.js';
import { generateProjectId } from '../utils/projectId.js';

const store = new ProjectStore();
const DEFAULTS: ProjectDefaults = { ...DEFAULT_AGENT_DEFAULTS };

export function registerProjectCommands(program: Command) {
  const project = program.command('project').description('Gerencia projetos MT5');

  project
    .command('init')
    .description('Cria projeto com terminal isolado (mtcli gerencia todos os caminhos)')
    .option('--id <id>', 'Nome do projeto (se não passar, será perguntado)')
    .option('--force', 'Recria se já existir', false)
    .action(async (opts) => {
      const existing = await store.show();
      const id = opts.id?.trim() || generateProjectId();
      if (!opts.id) {
        console.log(chalk.gray(`[project init] Gerado id automático: ${id}`));
      }
      if (existing.projects[id]) {
        const proceed = opts.force || (await promptYesNo(`Projeto "${id}" já existe. Recriar?`, false));
        if (!proceed) {
          console.log(chalk.yellow(`Projeto "${id}" já existe; nada foi alterado.`));
          await logProjectSummary(existing.projects[id]);
          return;
        }
      }
      const base = await ensureBaseTerminal();
      const install = await installTerminalForProject(id, base);
      const payload = {
        project: id,
        libs: install.libs,
        terminal: install.terminal,
        metaeditor: install.metaeditor,
        data_dir: install.dataDir,
        defaults: DEFAULTS,
      };
      const saved = await store.setProject(id, payload, true);
      console.log(chalk.green(`Projeto ${saved.project} inicializado com terminal isolado.`));
      await restartListenerInstance({ project: saved.project, profile: DEFAULTS.profile ?? undefined });
      await logProjectSummary(saved);
    });

  project
    .command('show')
    .description('Lista projetos configurados')
    .action(async () => {
      const file = await store.show();
      const entries = Object.entries(file.projects);
      if (entries.length === 0) {
        console.log('Nenhum projeto registrado. Use `mtcli project init` ou `project save`.');
        return;
      }
      console.log(chalk.bold('Projeto ativo:'), file.last_project || '(não definido)');
      for (const [id, info] of entries) {
        console.log(`\n${chalk.cyan(id)} -> libs=${info.libs}`);
        console.log(`  terminal: ${info.terminal || '(não definido)'}`);
        console.log(`  metaeditor: ${info.metaeditor || '(não definido)'}`);
        console.log(`  data_dir: ${info.data_dir || '(não definido)'}`);
        if (info.defaults) {
          console.log(`  defaults: ${JSON.stringify(info.defaults)}`);
        }
      }
    });

  project
    .command('save')
    .description('Cria ou recria um projeto com terminal isolado gerido pelo mtcli')
    .requiredOption('--id <id>', 'Nome do projeto')
    .option('--set-default', 'Torna este o projeto padrão', false)
    .action(async (opts) => {
      const base = await ensureBaseTerminal();
      const install = await installTerminalForProject(opts.id, base);
      const payload = {
        project: opts.id,
        libs: install.libs,
        terminal: install.terminal,
        metaeditor: install.metaeditor,
        data_dir: install.dataDir,
        defaults: DEFAULTS,
      };
      const saved = await store.setProject(opts.id, payload, opts.setDefault);
      console.log(chalk.green(`Projeto ${saved.project} salvo/recriado com terminal isolado.`));
      await restartListenerInstance({ project: saved.project, profile: saved.defaults?.profile ?? undefined });
      await logProjectSummary(saved);
    });

  const defaults = project.command('defaults').description('Configura opções padrão');

  defaults
    .command('show')
    .option('--id <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.id);
      console.log(info.defaults ? JSON.stringify(info.defaults, null, 2) : 'Sem defaults');
    });

  defaults
    .command('set')
    .option('--id <id>', 'Projeto alvo (default: last_project)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--subwindow <index>', '', (val) => parseInt(val, 10))
    .option('--indicator <name>')
    .option('--expert <name>')
    .option('--portable <flag>', 'true/false')
    .option('--profile <name>')
    .action(async (opts) => {
      const defaultsPayload: ProjectDefaults = {};
      if (opts.symbol) defaultsPayload.symbol = opts.symbol;
      if (opts.period) defaultsPayload.period = opts.period;
      if (Number.isInteger(opts.subwindow)) defaultsPayload.subwindow = opts.subwindow;
      if (opts.indicator !== undefined) defaultsPayload.indicator = opts.indicator;
      if (opts.expert !== undefined) defaultsPayload.expert = opts.expert;
      if (opts.profile) defaultsPayload.profile = opts.profile;
      if (opts.portable !== undefined) {
        defaultsPayload.portable = opts.portable === 'true' || opts.portable === true;
      }
      // Se id não for passado, usa last_project (store.useOrThrow resolve para o ativo)
      const target = opts.id ?? (await store.useOrThrow()).project;
      try {
        const updated = await store.updateDefaults(target, defaultsPayload);
        console.log(chalk.green(`Defaults atualizados para ${updated.project}`));
      } catch (err) {
        const file = await store.show();
        const known = Object.keys(file.projects);
        const hint = known.length ? `Projetos conhecidos: ${known.join(', ')}` : 'Nenhum projeto salvo. Use "mtcli project save --id <nome> --data-dir ... --libs ..."';
        throw new Error(`Projeto "${target}" não encontrado. ${hint}`);
      }
    });
}
