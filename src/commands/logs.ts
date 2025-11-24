import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';

const store = new ProjectStore();

type LogType = 'terminal' | 'metaeditor' | 'mql5' | 'tester';

function yyyymmdd(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
}

async function resolveLogPath(info: any, kind: LogType, date?: string): Promise<string> {
  if (!info.data_dir) throw new Error('data_dir não configurado.');
  const base = info.data_dir as string;
  const day = date || yyyymmdd(new Date());
  switch (kind) {
    case 'terminal':
      return path.join(base, 'Logs', `${day}.log`);
    case 'metaeditor':
      return path.join(base, 'Logs', 'metaeditor.log');
    case 'mql5':
      return path.join(base, 'MQL5', 'Logs', `${day}.log`);
    case 'tester':
      return path.join(base, 'Tester', 'logs', `${day}.log`);
    default:
      throw new Error('Tipo de log inválido');
  }
}

async function tailFile(file: string, lines: number): Promise<string[]> {
  if (!(await fs.pathExists(file))) {
    throw new Error(`Arquivo de log não encontrado: ${file}`);
  }
  const content = await fs.readFile(file, 'utf8');
  const arr = content.replace(/\r/g, '').split('\n');
  const slice = arr.slice(Math.max(0, arr.length - lines));
  return slice.filter((l) => l.length > 0);
}

export function registerLogsCommands(program: Command) {
  program
    .command('logs')
    .description('Mostra logs do MT5/MetaEditor/MQL5/Tester do projeto')
    .option('--type <terminal|metaeditor|mql5|tester>', 'Tipo de log', 'terminal')
    .option('--date <YYYYMMDD>', 'Data (default: hoje)', '')
    .option('--lines <n>', 'Qtd de linhas a mostrar', '200')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const kind = (opts.type as LogType) || 'terminal';
      const lines = Math.max(1, parseInt(opts.lines, 10) || 200);
      const logPath = await resolveLogPath(info, kind, opts.date || undefined);
      try {
        const tail = await tailFile(logPath, lines);
        console.log(chalk.cyan(`[logs] ${logPath} (últimas ${lines} linhas)`));
        tail.forEach((l) => console.log(l));
      } catch (err) {
        console.error(chalk.red(String(err instanceof Error ? err.message : err)));
      }
    });
}
