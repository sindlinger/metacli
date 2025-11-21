import { Command } from 'commander';
import chalk from 'chalk';
import {
  DEFAULT_AGENT_DATA_DIR,
  DEFAULT_AGENT_METAEDITOR,
  DEFAULT_AGENT_PROJECT_ID,
  DEFAULT_AGENT_TERMINAL,
  DEFAULT_AGENT_DEFAULTS,
  defaultAgentLibs,
} from '../config/agentDefaults.js';
import { ProjectDefaults, ProjectStore } from '../config/projectStore.js';
import { restartListenerInstance } from './listener.js';
import { logProjectSummary } from '../utils/projectSummary.js';

const store = new ProjectStore();

function parseBooleanFlag(value: unknown): boolean | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'boolean') return value;
  const normalized = String(value).trim().toLowerCase();
  if (['true', '1', 'yes', 'y'].includes(normalized)) return true;
  if (['false', '0', 'no', 'n'].includes(normalized)) return false;
  return undefined;
}

function buildDefaults(opts: Record<string, unknown>): ProjectDefaults {
  const defaults: ProjectDefaults = { ...DEFAULT_AGENT_DEFAULTS };
  if (typeof opts.symbol === 'string' && opts.symbol.length > 0) {
    defaults.symbol = opts.symbol;
  }
  if (typeof opts.period === 'string' && opts.period.length > 0) {
    defaults.period = opts.period;
  }
  if (typeof opts.profile === 'string' && opts.profile.length > 0) {
    defaults.profile = opts.profile;
  }
  if (typeof opts.subwindow === 'number' && Number.isFinite(opts.subwindow)) {
    defaults.subwindow = opts.subwindow;
  }
  const portable = parseBooleanFlag(opts.portable);
  if (portable !== undefined) {
    defaults.portable = portable;
  }
  return defaults;
}

export function registerInitCommand(program: Command) {
  program
    .command('init')
    .alias('up')
    .description('Liga o MT5 do agente, reinicia o listener e aplica defaults seguros')
    .option('--project <id>', 'Nome do projeto padrão', DEFAULT_AGENT_PROJECT_ID)
    .option('--terminal <path>', 'Caminho do terminal64.exe')
    .option('--metaeditor <path>', 'Caminho do MetaEditor64.exe')
    .option('--data-dir <path>', 'Pasta de dados do MT5 (contém MQL5)')
    .option('--libs <path>', 'Pasta MQL5\\Libraries com as DLLs do projeto')
    .option('--symbol <symbol>', 'Símbolo padrão dos comandos')
    .option('--period <period>', 'Período padrão (H1, M15, etc.)')
    .option('--subwindow <index>', 'Subjanela padrão do indicador', (val) => parseInt(val, 10))
    .option('--profile <name>', 'Perfil padrão do terminal')
    .option('--portable <flag>', 'Define portable true/false (default herdado)')
    .action(async (opts) => {
      const project = opts.project || DEFAULT_AGENT_PROJECT_ID;
      const dataDir = opts.dataDir || DEFAULT_AGENT_DATA_DIR;
      const libs = opts.libs || defaultAgentLibs(dataDir);
      if (!libs) {
        throw new Error('libs não definidos. Informe --libs ou configure DEFAULT_AGENT_DATA_DIR corretamente.');
      }
      const payload = {
        project,
        libs,
        terminal: opts.terminal || DEFAULT_AGENT_TERMINAL,
        metaeditor: opts.metaeditor || DEFAULT_AGENT_METAEDITOR,
        data_dir: dataDir,
        defaults: buildDefaults(opts),
      };
      const saved = await store.setProject(project, payload, true);
      console.log(chalk.green(`[init] Projeto ${project} configurado.`));
      await restartListenerInstance({ project: saved.project, profile: saved.defaults?.profile ?? undefined });
      await logProjectSummary(saved);
    });
}
