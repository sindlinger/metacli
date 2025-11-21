import chalk from 'chalk';
import { ProjectInfo } from '../config/projectStore.js';
import { printLatestLogFromDataDir } from './logs.js';

export async function logProjectSummary(info: ProjectInfo, logLines = 20) {
  console.log(chalk.bold(`[project ${info.project}]`));
  console.log(`  libs: ${info.libs}`);
  console.log(`  terminal: ${info.terminal || '(não definido)'}`);
  console.log(`  metaeditor: ${info.metaeditor || '(não definido)'}`);
  console.log(`  data_dir: ${info.data_dir || '(não definido)'}`);
  if (info.defaults) {
    console.log(`  defaults: ${JSON.stringify(info.defaults)}`);
  }
  if (info.data_dir) {
    await printLatestLogFromDataDir(info.data_dir, logLines);
  }
}
