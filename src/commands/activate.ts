import { Command } from 'commander';
import chalk from 'chalk';
import path from 'path';
import fs from 'fs-extra';
import { ProjectStore, ProjectDefaults, repoRoot } from '../config/projectStore.js';
import { softActivate } from './terminal.js';

const store = new ProjectStore();

export function registerActivateCommand(program: Command) {
  program
    .command('activate')
    .description('Reaplica configs/credenciais/templates e relança o terminal do projeto')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      if (!opts.project) {
        throw new Error('Informe o --project.');
      }
      let info: any;
      try {
        info = await store.useOrThrow(opts.project);
      } catch {
        const dataDir = path.resolve(repoRoot(), 'projects', 'terminals', opts.project);
        const terminal = path.join(dataDir, 'terminal64.exe');
        if (!(await fs.pathExists(terminal))) {
          throw new Error(
            `Projeto "${opts.project}" não registrado e terminal64.exe não encontrado em ${terminal}. Rode mtcli init ${opts.project} primeiro.`
          );
        }
        info = {
          project: opts.project,
          data_dir: dataDir,
          terminal,
          metaeditor: path.join(dataDir, 'MetaEditor64.exe'),
          libs: path.join(dataDir, 'MQL5', 'Libraries'),
          defaults: {
            symbol: 'EURUSD',
            period: 'M1',
            subwindow: 1,
            indicator: 'Examples\\ZigZag',
            expert: 'Examples\\MACD\\MACD Sample',
            portable: true,
            profile: null,
          },
        } as any;
        await store.setProject(
          info.project,
          {
            project: info.project,
            data_dir: info.data_dir,
            terminal: info.terminal,
            metaeditor: info.metaeditor,
            libs: info.libs,
            defaults: info.defaults,
          },
          true
        );
        console.log(chalk.gray(`[activate] Projeto "${info.project}" registrado a partir de ${dataDir}.`));
      }

      const defaults: ProjectDefaults = info.defaults || {
        symbol: 'EURUSD',
        period: 'M1',
        subwindow: 1,
        indicator: 'Examples\\ZigZag',
        expert: 'Examples\\MACD\\MACD Sample',
        portable: true,
        profile: null,
      };

      await softActivate(info, defaults);
    });
}
