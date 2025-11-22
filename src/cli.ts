import { Command } from 'commander';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { registerProjectCommands } from './commands/project.js';
import { registerListenerCommands } from './commands/listener.js';
import { registerChartCommands, registerIndicatorCommands, registerExpertCommands } from './commands/chart.js';
import { registerTesterCommands } from './commands/tester.js';
import { registerEditorCommands } from './commands/editor.js';
import { registerDllCommands } from './commands/dll.js';
import { registerUtilsCommands } from './commands/utils.js';
import { registerConfigCommands } from './commands/config.js';

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

  registerProjectCommands(program);
  registerListenerCommands(program);
  registerChartCommands(program);
  registerIndicatorCommands(program);
  registerExpertCommands(program);
  registerTesterCommands(program);
  registerEditorCommands(program);
  registerDllCommands(program);
  registerUtilsCommands(program);
  registerConfigCommands(program);

  program.addHelpText('after', '\nExemplo rápido:\n  mtcli project show\n  mtcli listener start\n  mtcli chart indicator attach --symbol EURUSD --period H1 --indicator WaveSpecZZ_Project/WaveSpecZZ_1.1.0-gpuopt');

  try {
    await program.parseAsync(process.argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(chalk.red(message));
    process.exit(1);
  }
}

main();
