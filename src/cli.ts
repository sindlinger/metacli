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
import { registerVerifyCommands } from './commands/verify.js';
import { registerLogsCommands } from './commands/logs.js';
import { registerDucascopyCommands } from './commands/ducascopy.js';
import { registerCompileCommands } from './commands/compile.js';
import { registerHealthCommand } from './commands/health.js';
import { registerOpenTerminalCommand } from './commands/openTerminal.js';
import { registerCloseTerminalCommand } from './commands/closeTerminal.js';
import { registerLaunchCmd } from './commands/launchCmd.js';
import { registerProjectCommands } from './commands/project.js';
import { registerPingCommand } from './commands/ping.js';
import { registerActivateCommand } from './commands/activate.js';
import { registerDevCommand } from './commands/dev.js';

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
  program.configureHelp({ sortSubcommands: false });

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
  registerChartCommands(program);
  registerIndicatorCommands(program);
  registerExpertCommands(program);
  registerTesterCommands(program);
  registerCompileCommands(program);
  registerHealthCommand(program);
  registerOpenTerminalCommand(program);
  registerCloseTerminalCommand(program);
  registerPingCommand(program);
  registerLaunchCmd(program);
  registerActivateCommand(program);
  registerDevCommand(program);
  registerGpuCommands(program);
  registerUtilsCommands(program);
  registerProjectCommands(program);
  registerVerifyCommands(program);
  registerConfigCommands(program);
  registerLogsCommands(program);
  registerDucascopyCommands(program);

  program.addHelpText(
    'after',
    '\n' +
      chalk.bold('Sugestões rápidas:') +
      '\n  mtcli init' +
      '\n  mtcli compile -i \\Indicators\\Examples\\ZigZag' +
      '\n  mtcli run -i Examples\\ZigZag' +
      '\n  mtcli run -e Moving_Average -v' +
      '\n  mtcli chart capture' +
      '\n  mtcli logs --type terminal' +
      '\n  mtcli verify' +
      '\n  mtcli tester quick --expert MyEA --symbol EURUSD --period H1' +
      '\n' +
      chalk.bold('\nExemplos por comando:') +
      '\n  init                mtcli init' +
      '\n  compile             mtcli compile -e MyEA.mq5' +
      '\n  run                 mtcli run -e MyEA --visual' +
      '\n  chart capture       mtcli chart capture --symbol EURUSD --period H1' +
      '\n  chart objects list  mtcli chart objects list' +
      '\n  logs                mtcli logs --type mql5 --lines 100' +
      '\n  verify              mtcli verify' +
      '\n  config paths        mtcli config paths' +
      '\n  config ls           mtcli config ls --scope mql5 --path Experts' +
      '\n  tester quick        mtcli tester quick --expert MyEA --symbol EURUSD --period H1' +
      '\n  gpu link            mtcli gpu link --config Release' +
      '\n'
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
