import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore, ProjectDefaults } from '../config/projectStore.js';
import { restartListenerInstance } from './listener.js';

const store = new ProjectStore();
const DEFAULT_DATA_DIR = 'C:/Users/pichau/AppData/Roaming/MetaQuotes/Terminal/72D7079820AB4E374CDC07CD933C3265';
const DEFAULT_TERMINAL = 'C:/Dukascopy MetaTrader 5/terminal64.exe';
const DEFAULT_METAEDITOR = 'C:/Dukascopy MetaTrader 5/MetaEditor64.exe';
const DEFAULT_LIBS = `${DEFAULT_DATA_DIR}/MQL5/Libraries`;
const DEFAULTS: ProjectDefaults = {
  symbol: 'EURUSD',
  period: 'H1',
  subwindow: 1,
  profile: 'Default',
  portable: false,
};

export function registerProjectCommands(program: Command) {
  const project = program.command('project').description('Gerencia projetos MT5');

  project
    .command('init')
    .description('Cria projeto usando os caminhos padrão deste ambiente')
    .option('--id <id>', 'Nome do projeto', 'dukas-cli')
    .option('--no-start-listener', 'Não reinicia o terminal automaticamente')
    .action(async (opts) => {
      const payload = {
        project: opts.id,
        libs: DEFAULT_LIBS,
        terminal: DEFAULT_TERMINAL,
        metaeditor: DEFAULT_METAEDITOR,
        data_dir: DEFAULT_DATA_DIR,
        defaults: DEFAULTS,
      };
      const saved = await store.setProject(opts.id, payload, true);
      console.log(chalk.green(`Projeto ${saved.project} inicializado com caminhos padrão.`));
      if (opts.startListener === false) {
        console.log('Listener não reiniciado (--no-start-listener).');
      } else {
        await restartListenerInstance({ project: saved.project, profile: DEFAULTS.profile ?? undefined });
      }
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
    .description('Atualiza ou cria um projeto')
    .requiredOption('--id <id>', 'Nome do projeto')
    .option('--libs <path>', 'Caminho para MQL5\\Libraries')
    .option('--terminal <path>', 'terminal64.exe')
    .option('--metaeditor <path>', 'metaeditor64.exe')
    .option('--data-dir <path>', 'Pasta de dados (contém MQL5)')
    .option('--set-default', 'Torna este o projeto padrão', false)
    .option('--launch-listener', 'Reinicia listener ao salvar', false)
    .action(async (opts) => {
      const payload = {
        project: opts.id,
        libs: opts.libs,
        terminal: opts.terminal,
        metaeditor: opts.metaeditor,
        data_dir: opts.dataDir || opts['dataDir'],
      };
      if (!payload.libs) {
        throw new Error('Informe --libs');
      }
      const saved = await store.setProject(opts.id, payload, opts.setDefault);
      console.log(chalk.green(`Projeto ${saved.project} salvo.`));
      if (opts.launchListener && payload.terminal && payload.data_dir) {
        await restartListenerInstance({ project: saved.project, profile: saved.defaults?.profile ?? undefined });
      }
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
    .requiredOption('--id <id>', 'Projeto alvo')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--subwindow <index>', '', (val) => parseInt(val, 10))
    .option('--indicator <name>')
    .option('--portable <flag>', 'true/false')
    .option('--profile <name>')
    .action(async (opts) => {
      const defaultsPayload: ProjectDefaults = {};
      if (opts.symbol) defaultsPayload.symbol = opts.symbol;
      if (opts.period) defaultsPayload.period = opts.period;
      if (Number.isInteger(opts.subwindow)) defaultsPayload.subwindow = opts.subwindow;
      if (opts.indicator !== undefined) defaultsPayload.indicator = opts.indicator;
      if (opts.profile) defaultsPayload.profile = opts.profile;
      if (opts.portable !== undefined) {
        defaultsPayload.portable = opts.portable === 'true' || opts.portable === true;
      }
      const updated = await store.updateDefaults(opts.id, defaultsPayload);
      console.log(chalk.green(`Defaults atualizados para ${updated.project}`));
    });
}
