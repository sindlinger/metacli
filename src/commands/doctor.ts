import { Command } from 'commander';
import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { isListenerRunning } from './listener.js';
import { toWslPath } from '../utils/paths.js';

const store = new ProjectStore();

export function registerDoctorCommands(program: Command) {
  program
    .command('doctor')
    .description('Checagem rápida do ambiente MTCLI/MT5')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const checks: Array<[string, boolean, string]> = [];

      // terminal
      checks.push(['terminal configurado', Boolean(info.terminal), info.terminal || '']);
      if (info.terminal) checks.push(['terminal existe', fs.existsSync(info.terminal), info.terminal]);

      // data_dir
      checks.push(['data_dir configurado', Boolean(info.data_dir), info.data_dir || '']);
      if (info.data_dir) checks.push(['data_dir existe', fs.existsSync(info.data_dir), info.data_dir]);

      // libs
      if (info.libs) checks.push(['libs existe', fs.existsSync(info.libs), info.libs]);

      // listener
      const listenerOk = info.terminal ? await isListenerRunning(info.terminal) : false;
      checks.push(['listener rodando', listenerOk, info.terminal ? toWslPath(info.terminal) : '']);

      console.log(chalk.cyan('doctor:')); 
      for (const [label, ok, note] of checks) {
        const color = ok ? chalk.green : chalk.red;
        console.log(`  ${color(ok ? 'OK' : 'FAIL')} ${label} ${note ? `(${note})` : ''}`);
      }

      const fails = checks.filter(([, ok]) => !ok);
      if (fails.length === 0) {
        console.log(chalk.green('\nTudo certo.'));
      } else {
        console.log(chalk.yellow(`\nPendências: ${fails.length}. Ajuste os FAIL acima.`));
      }
    });
}
