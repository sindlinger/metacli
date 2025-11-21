import path from 'path';
import { ProjectDefaults } from './projectStore.js';

export const DEFAULT_AGENT_PROJECT_ID = 'agent-terminal';
export const DEFAULT_AGENT_DATA_DIR =
  'C:/Users/pichau/AppData/Roaming/MetaQuotes/Terminal/72D7079820AB4E374CDC07CD933C3265';
export const DEFAULT_AGENT_TERMINAL = 'C:/Dukascopy MetaTrader 5/terminal64.exe';
export const DEFAULT_AGENT_METAEDITOR = 'C:/Dukascopy MetaTrader 5/MetaEditor64.exe';

export const DEFAULT_AGENT_DEFAULTS: ProjectDefaults = {
  symbol: 'EURUSD',
  period: 'H1',
  subwindow: 1,
  profile: 'Default',
  portable: false,
};

export function defaultAgentLibs(dataDir: string = DEFAULT_AGENT_DATA_DIR): string {
  return path.join(dataDir, 'MQL5', 'Libraries');
}
