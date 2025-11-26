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
import { registerProjectCommands } from './commands/project.js';
import { registerActivateCommand } from './commands/activate.js';
import { registerDevCommand } from './commands/dev.js';
import { registerTerminalControlCommands } from './commands/terminalControl.js';

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
  registerActivateCommand(program);
  registerTerminalControlCommands(program);
  registerDevCommand(program);
  registerGpuCommands(program);
  registerUtilsCommands(program);
  registerProjectCommands(program);
  registerVerifyCommands(program);
  registerConfigCommands(program);
  registerLogsCommands(program);
  registerDucascopyCommands(program);

  // Agrupamento em 4 sessões coloridas
  const groupDefs: Array<{ key: string; title: string; color: (s: string) => string; commands: string[] }> = [
    { key: 'starter', title: 'Início/Plataforma', color: chalk.gray, commands: ['init', 'activate', 'config'] },
    {
      key: 'dev',
      title: 'Dev (Indicadores/EA/Scripts)',
      color: chalk.green,
      commands: ['dev', 'compile', 'tester', 'watch', 'indicator', 'expert', 'chart', 'snapshot', 'events', 'logs', 'inputs'],
    },
    {
      key: 'mt5',
      title: 'MT5 Operação',
      color: chalk.cyan,
      commands: ['events', 'globals', 'objects', 'chart', 'trade', 'logs', 'verify', 'config', 'ping', 'launch-cmd', 'ducascopy', 'copy', 'terminal'],
    },
    {
      key: 'maw',
      title: 'Men at work',
      color: (chalk as any).keyword ? (chalk as any).keyword('orange') : chalk.yellow,
      commands: ['gpu', 'health', 'listener', 'project', 'utils'],
    },
  ];

  const existing = new Set(program.commands.map((c) => c.name()));
  const fmtList = (list: string[], color: (s: string) => string) => color(list.map((n) => `- ${n}`).join('\n  '));

  const blocks: string[] = [];
  for (const g of groupDefs) {
    const present = g.commands.filter((c) => existing.has(c));
    if (present.length) blocks.push(g.color(g.title) + '\n  ' + fmtList(present, g.color));
  }
  // comandos que sobraram
  const grouped = new Set(groupDefs.flatMap((g) => g.commands));
  const restList = Array.from(existing).filter((n) => !grouped.has(n) && n !== 'help').sort();
  if (restList.length) {
    blocks.push(chalk.white('Outros') + '\n  ' + fmtList(restList, chalk.white));
  }

  if (blocks.length) {
    program.addHelpText('beforeAll', '\n' + blocks.join('\n\n') + '\n');
  }

  // Comandos de atalho para listar apenas um grupo (mt5 / maw)
  const renderGroup = (key: string) => {
    const g = groupDefs.find((g) => g.key === key);
    if (!g) return () => {};
    return () => {
      const present = g.commands.filter((c) => existing.has(c));
      console.log(g.color(g.title));
      present.forEach((name) => {
        const cmd = program.commands.find((c) => c.name() === name);
        const desc = cmd?.description() ?? '';
        console.log(`  - ${name}  ${desc}`);
      });
    };
  };

  program.command('mt5').description('Mostra somente os comandos de MT5 Operação').action(renderGroup('mt5'));
  program.command('maw').description('Mostra somente os comandos Men at work').action(renderGroup('maw'));

  program.addHelpText(
    'after',
    '\n' +
      chalk.bold('Sugestões rápidas:') +
      '\n  mtcli init' +
      '\n  mtcli activate --project <id>' +
      '\n  mtcli terminal start --project <id>' +
      '\n  mtcli dev compile -i Indicators\\Examples\\ZigZag' +
      '\n  mtcli dev watch -i Indicators\\ReversalWave_OpenCL' +
      '\n  mtcli dev indicator buffers -n ReversalWave_OpenCL' +
      '\n  mtcli logs --type terminal' +
      '\n  mtcli verify' +
      '\n  mtcli tester quick --expert MyEA --symbol EURUSD --period H1' +
      '\n' +
      chalk.bold('\nExemplos por comando:') +
      '\n  init                mtcli init' +
      '\n  activate            mtcli activate --project project-XYZ' +
      '\n  terminal start      mtcli terminal start --project project-XYZ' +
      '\n  dev compile         mtcli dev compile -e MyEA.mq5' +
      '\n  dev watch           mtcli dev watch -i Indicators\\MyInd' +
      '\n  dev indicator add   mtcli dev indicator add -n MyInd -s EURUSD -p M1' +
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
