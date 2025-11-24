import path from 'path';
import { ProjectDefaults } from './projectStore.js';

export const DEFAULT_AGENT_PROJECT_ID = 'mql5-terminal';
// Default terminal the agent must control (usuário configurou em C:\mql5)
export const DEFAULT_AGENT_DATA_DIR =
  'C:/Users/pichau/AppData/Roaming/MetaQuotes/Terminal/80DC949B6D3DF1DE0B919D359660D2E7';
export const DEFAULT_AGENT_TERMINAL = 'C:/mql5/terminal64.exe';
export const DEFAULT_AGENT_METAEDITOR = 'C:/mql5/MetaEditor64.exe';

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
