import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';

const store = new ProjectStore();

async function tailFile(file: string, lines = 200) {
  if (!(await fs.pathExists(file))) return [] as string[];
  const content = await fs.readFile(file, 'utf8');
  const arr = content.replace(/\r/g, '').split('\n');
  return arr.slice(-lines).filter(Boolean);
}

async function tailLogs(info: any, lines: number, filter?: string) {
  if (!info.data_dir) throw new Error('data_dir não configurado para o projeto.');
  const logsDir = path.join(info.data_dir, 'Logs');
  if (!(await fs.pathExists(logsDir))) {
    console.log(chalk.yellow(`Logs não encontrados em ${logsDir}`));
    return;
  }
  const files = (await fs.readdir(logsDir))
    .filter((f) => f.toLowerCase().endsWith('.log'))
    .sort()
    .slice(-3) // últimos dias
    .map((f) => path.join(logsDir, f));
  const all: string[] = [];
  for (const f of files) {
    const tail = await tailFile(f, lines);
    all.push(...tail.map((line) => `${path.basename(f)}: ${line}`));
  }
  const filtered = filter ? all.filter((l) => l.toLowerCase().includes(filter.toLowerCase())) : all;
  filtered.slice(-lines).forEach((l) => console.log(l));
}

export function registerEventsCommands(program: Command) {
  const events = program.command('events').description('Logs, ping e mensagens de depuração');

  events
    .command('tail')
    .option('-n, --lines <n>', 'Linhas', (v) => parseInt(v, 10))
    .option('-f, --filter <text>', 'Filtro (contém)')
    .option('--follow', 'Segue em tempo real', false)
    .option('--errors', 'Filtra erros (error/critical/fail)', false)
    .option('--project <id>', '(LEGADO; evite, usa ativo)')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const n = Number.isFinite(opts.lines) ? opts.lines : 200;
      const flt = opts.errors ? 'error' : opts.filter;
      await tailLogs(info, n, flt);
      if (opts.follow) {
        console.log(chalk.gray('[events] follow ativo (Ctrl+C para parar)'));
        const logsDir = info.data_dir ? path.join(info.data_dir, 'Logs') : '';
        if (!logsDir || !(await fs.pathExists(logsDir))) return;
        fs.watch(logsDir, async () => {
          await tailLogs(info, n, flt);
        });
        // manter vivo
        await new Promise(() => {});
      }
    });

  events
    .command('send')
    .requiredOption('--text <msg>', 'Mensagem de depuração')
    .option('--project <id>', '(LEGADO; evite, usa ativo)')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendListenerCommand(info, 'DEBUG_MSG', [opts.text], { timeoutMs: 4000 });
      console.log(chalk.green('[events] mensagem enviada')); 
    });

  events
    .command('ping')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendListenerCommand(info, 'PING', [], { timeoutMs: 4000, ensureRunning: true, allowRestart: false });
      console.log(chalk.green('[events] PING ok'));
    });

  events
    .command('clear')
    .description('Apaga logs do data_dir/Logs')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado para o projeto.');
      const logsDir = path.join(info.data_dir, 'Logs');
      if (!(await fs.pathExists(logsDir))) {
        console.log(chalk.yellow(`Logs não encontrados em ${logsDir}`));
        return;
      }
      const files = await fs.readdir(logsDir);
      for (const f of files) {
        await fs.remove(path.join(logsDir, f));
      }
      console.log(chalk.green('[events] logs apagados'));
    });
}
