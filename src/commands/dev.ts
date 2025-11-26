import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { ensureHealth } from '../utils/listenerGuard.js';
import { registerCompileCommands } from './compile.js';
import { registerTesterCommands } from './tester.js';
import { registerWatchCommands } from './watch.js';
import { registerIndicatorCommands, registerExpertCommands, registerChartCommands } from './chart.js';
import { registerSnapshotCommands } from './snapshot.js';
import { registerEventsCommands } from './events.js';
import { registerLogsCommands } from './logs.js';

const store = new ProjectStore();

/**
 * Agrupa comandos de desenvolvimento (indicadores/EAs/scripts) sob o prefixo "dev".
 * Exemplo: mtcli dev compile -i Indicators\MyInd.mq5
 */
export function registerDevCommand(program: Command) {
  const dev = program.command('dev').description('Modo desenvolvimento (indicadores/EAs/scripts)');

  // Autorização: exige projeto ativo e health recente registrado pelo activate
  dev.hook('preAction', async (_thisCmd, actionCmd) => {
    try {
      const info = await store.useOrThrow((actionCmd as any).opts().project || undefined);
      await ensureHealth(info, 60000);
    } catch (err: any) {
      console.error(
        chalk.red(
          `[dev] Health ausente/expirado. Rode mtcli activate para registrar e tente novamente. Detalhe: ${err?.message || err}`
        )
      );
      process.exit(1);
    }
  });

  registerCompileCommands(dev);
  registerTesterCommands(dev);
  registerWatchCommands(dev);
  registerIndicatorCommands(dev);
  registerExpertCommands(dev);
  registerChartCommands(dev);
  registerSnapshotCommands(dev);
  registerEventsCommands(dev);
  registerLogsCommands(dev);
}
