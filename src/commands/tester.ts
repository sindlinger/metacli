import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, repoRoot } from '../config/projectStore.js';
import { normalizePath, toWinPath, toWslPath } from '../utils/paths.js';
import { runCommand } from '../utils/shell.js';

const store = new ProjectStore();
const DEFAULT_TESTER_INI = path.join(repoRoot(), 'tester_visual.ini');

export function registerTesterCommands(program: Command) {
  const tester = program.command('tester').description('Opera o Strategy Tester');

  tester
    .command('status')
    .description('Mostra estado básico do tester')
    .action(() => {
      console.log('tester status: consulte os relatórios/arquivos do MT5 (placeholder).');
    });

  tester
    .command('run')
    .description('Executa o terminal no modo Strategy Tester usando um arquivo .ini existente')
    .option('--config <path>', `Arquivo .ini (default: ${DEFAULT_TESTER_INI})`)
    .option('--project <id>', 'Projeto configurado')
    .option('--wait', 'Aguardar o terminal encerrar em vez de destacar', false)
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal) {
        throw new Error('terminal64.exe não configurado no projeto.');
      }
      if (!info.data_dir) {
        throw new Error('Defina data_dir no projeto (mtcli project save --data-dir ...).');
      }
      const iniPath = normalizePath(opts.config || DEFAULT_TESTER_INI);
      if (!(await fs.pathExists(iniPath))) {
        throw new Error(`Arquivo .ini não encontrado: ${iniPath}`);
      }
      const args = [`/config:${toWinPath(iniPath)}`, `/datapath:${toWinPath(info.data_dir)}`];
      const exe = toWslPath(info.terminal);
      const wait = Boolean(opts.wait);
      await runCommand(exe, args, { stdio: wait ? 'inherit' : 'ignore', detach: !wait });
      console.log(chalk.green(`Strategy Tester iniciado com ${iniPath}`));
    });
}
