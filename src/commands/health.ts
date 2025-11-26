import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';
import { ensureCommandListenerCompiled, killTerminalIfRunning, killTerminalWindows, killTerminalByDatapath, startTerminalWindows } from '../utils/commandListener.js';
import { ensureCommandListenerStartup } from '../utils/factoryAssets.js';
import { deployFactoryConfig, deployFactoryTemplates } from '../utils/factoryAssets.js';
import path from 'path';

const store = new ProjectStore();

export function registerHealthCommand(program: Command) {
  program
    .command('health')
    .description('Diagnostica CommandListener; opcionalmente repara e reanexa.')
    .option('--auto', 'Tenta reparar (recompilar e reanexar) sem reiniciar terminal', false)
    .option('--restart', 'Tenta reiniciar terminal do projeto se continuar offline', false)
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir || !info.terminal) {
        throw new Error('data_dir/terminal não configurados. Rode mtcli init.');
      }
      const dataDir = info.data_dir;
      const terminal = info.terminal;

      // Passo 1: tentar PING direto
      const ping = async () =>
        sendListenerCommand(info, 'PING', [], { timeoutMs: 3000, ensureRunning: false, allowRestart: false });
      const tryPing = async () => {
        try {
          await ping();
          console.log(chalk.green('[health] PING OK (CommandListener online).'));
          return true;
        } catch (err) {
          return false;
        }
      };

      if (await tryPing()) return;

      if (!opts.auto) {
        console.log(
          chalk.yellow(
            '[health] CommandListener offline. Rode com --auto para recompilar/reattach, ou --restart para reiniciar terminal.'
          )
        );
        return;
      }

      // Passo 2: reparar sem reiniciar (recompilar + garantir StartUp + ATTACH)
      console.log(chalk.gray('[health] reparando: compilando CommandListenerEA e garantindo StartUp...'));
      await deployFactoryConfig(dataDir);
      await deployFactoryTemplates(dataDir);
      await ensureCommandListenerStartup(dataDir, info.defaults);
      await ensureCommandListenerCompiled(dataDir);

      // Tentar ATTACH do EA via listener, mas se offline, isso falhará; tentamos de qualquer jeito.
      try {
        await sendListenerCommand(info, 'ATTACH_EA_FULL', ['EURUSD', 'M1', 'CommandListener', 'Default.tpl', ''], {
          timeoutMs: 4000,
          ensureRunning: false,
          allowRestart: false,
        });
      } catch {
        // ignorar; objetivo é só tentar
      }

      if (await tryPing()) return;

      if (!opts.restart) {
        console.log(
          chalk.yellow(
            '[health] Ainda offline após reparo sem restart. Rode novamente com --restart para reiniciar só o terminal do projeto.'
          )
        );
        return;
      }

      // Passo 3: restart controlado do terminal do projeto
      console.log(chalk.gray('[health] reiniciando terminal do projeto...'));
      await killTerminalIfRunning(terminal).catch(() => {});
      await killTerminalWindows(terminal).catch(() => {});
      await killTerminalByDatapath(dataDir).catch(() => {});
      await startTerminalWindows(terminal, dataDir).catch(() => {});

      // Pequeno atraso para terminal subir
      await new Promise((r) => setTimeout(r, 4000));

      if (await tryPing()) return;

      throw new Error(
        'CommandListener continua offline após restart. Abra o terminal do projeto manualmente e verifique se o EA está anexado.'
      );
    });
}
