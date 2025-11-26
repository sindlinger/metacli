import { Command } from 'commander';
import path from 'path';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { toWslPath } from '../utils/paths.js';
import { execa } from 'execa';

async function toWin(p: string): Promise<string> {
  if (process.platform === 'win32') return p;
  const { stdout } = await execa('wslpath', ['-w', p]);
  return stdout.trim();
}

const store = new ProjectStore();

export function registerLaunchCmd(program: Command) {
  program
    .command('launch-cmd')
    .description('Mostra a linha exata (PowerShell) para abrir o terminal do projeto no Windows')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal || !info.data_dir) {
        throw new Error('terminal/data_dir não configurados. Rode mtcli init primeiro.');
      }
      const exe = await toWin(info.terminal);
      const dp = await toWin(info.data_dir);
      const cfg = await toWin(path.join(info.data_dir, 'Config', 'common.ini'));
      const args = [
        "'/portable'",
        `'/datapath:${dp}'`,
        `'/config:${cfg}'`,
        "'/expert:CommandListenerEA'",
        "'/symbol:EURUSD'",
        "'/period:M1'",
        "'/template:mtcli-default.tpl'",
      ].join(', ');
      const psLine = `Start-Process -FilePath '${exe}' -WorkingDirectory '${path.dirname(exe)}' -ArgumentList ${args}`;

      console.log(chalk.cyan('Cole no PowerShell do Windows:'));
      console.log(psLine);
      console.log(chalk.gray('\nSe preferir, crie um atalho/.ps1 com esse comando.'));
    });
}
