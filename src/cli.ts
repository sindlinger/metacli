import { Command } from 'commander';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { registerInitCommand } from './commands/init.js';
import { registerChartCommands, registerIndicatorCommands, registerExpertCommands } from './commands/chart.js';
import { registerTesterCommands } from './commands/tester.js';
import { registerGpuCommands } from './commands/gpu.js';
import { registerUtilsCommands } from './commands/utils.js';
import { registerGlobalsCommands } from './commands/globals.js';
import { registerEventsCommands } from './commands/events.js';
import { registerObjectsCommands } from './commands/objects.js';
import { registerInputsCommands } from './commands/inputs.js';
import { registerWatchCommands } from './commands/watch.js';
import { registerRawCommands } from './commands/raw.js';
import { registerSnapshotCommands } from './commands/snapshot.js';
import { registerTradeCommands } from './commands/trade.js';
import { registerCopyCommands } from './commands/rawcopy.js';
import { registerConfigCommands } from './commands/terminal.js';

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
  registerGlobalsCommands(program);
  registerEventsCommands(program);
  registerObjectsCommands(program);
  registerInputsCommands(program);
  registerWatchCommands(program);
  registerRawCommands(program);
  registerSnapshotCommands(program);
  registerTradeCommands(program);
  registerCopyCommands(program);
  registerConfigCommands(program);
  registerChartCommands(program);
  registerIndicatorCommands(program);
  registerExpertCommands(program);
  registerTesterCommands(program);
  registerGpuCommands(program);
  registerUtilsCommands(program);
  registerConfigCommands(program);

  program.addHelpText(
    'after',
    '\nAtalhos rápidos:\n' +
      '  mtcli init\n' +
      '  mtcli compile -i \\Indicators\\Examples\\ZigZag\n' +
      '  mtcli run -i Examples\\ZigZag\n' +
      '  mtcli run -e Moving_Average -v\n' +
      '  mtcli chart capture\n' +
      '  mtcli globals list\n' +
      '  mtcli events tail --errors --follow\n' +
      '  mtcli watch --file Foo.mq5 --indicator Foo\n' +
      '  mtcli tester quick --expert MyEA --symbol EURUSD --period H1\n'
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
