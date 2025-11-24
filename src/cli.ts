import { Command } from 'commander';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { registerProjectCommands } from './commands/project.js';
import { registerInitCommand } from './commands/init.js';
import { registerChartCommands, registerIndicatorCommands, registerExpertCommands } from './commands/chart.js';
import { registerTesterCommands } from './commands/tester.js';
import { registerEditorCommands } from './commands/editor.js';
import { registerDllCommands } from './commands/dll.js';
import { registerUtilsCommands } from './commands/utils.js';
import { registerConfigCommands } from './commands/config.js';
import { registerQuickCommands } from './commands/quick.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageJsonPath = path.resolve(__dirname, '..', 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

async function main() {
  const program = new Command();
  program
    .name('mtcli')
    .description('CLI modular para automatizar tarefas do MetaTrader 5 (CommandListenerEA)')
    .version(packageJson.version || '0.0.0')
    .showHelpAfterError();

  registerInitCommand(program);
  registerQuickCommands(program);
  registerProjectCommands(program);
  registerChartCommands(program);
  registerIndicatorCommands(program);
  registerExpertCommands(program);
  registerTesterCommands(program);
  registerEditorCommands(program);
  registerDllCommands(program);
  registerUtilsCommands(program);
  registerConfigCommands(program);

  program.addHelpText(
    'after',
    '\nAtalhos rápidos:\n' +
      '  mtcli init\n' +
      '  mtcli compile -i \\Indicators\\Examples\\ZigZag\n' +
      '  mtcli run -i Examples\\ZigZag\n' +
      '  mtcli run -e Moving_Average -v\n' +
      '  mtcli chart capture\n'
  );

  try {
    await program.parseAsync(process.argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(chalk.red(message));
    process.exit(1);
  }
}

main();
