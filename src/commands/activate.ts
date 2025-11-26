import { Command } from 'commander';
import chalk from 'chalk';
import { ProjectStore, ProjectDefaults } from '../config/projectStore.js';
import { softActivate } from './terminal.js';

const store = new ProjectStore();

export function registerActivateCommand(program: Command) {
  program
    .command('activate')
    .description('Reaplica configs/credenciais/templates e relança o terminal do projeto existente')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const defaults: ProjectDefaults = info.defaults || {
        symbol: 'EURUSD',
        period: 'M1',
        subwindow: 1,
        indicator: 'Examples\\ZigZag',
        expert: 'Examples\\MACD\\MACD Sample',
        portable: true,
        profile: null,
      };
      await softActivate(info as any, defaults);
    });
}
