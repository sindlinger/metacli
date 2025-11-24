import { Command } from 'commander';
import path from 'path';
import chalk from 'chalk';
import fs from 'fs-extra';
import { ProjectStore } from '../config/projectStore.js';
import { normalizePath } from '../utils/paths.js';
import { commonIniPath } from './terminal.js';

const store = new ProjectStore();

type VerifyKind = 'file' | 'dir';
interface VerifyItem {
  label: string;
  path: string;
  kind: VerifyKind;
}

async function statExists(target: string, type: VerifyKind | 'any' = 'any') {
  if (!(await fs.pathExists(target))) return false;
  if (type === 'any') return true;
  const st = await fs.stat(target);
  return type === 'file' ? st.isFile() : st.isDirectory();
}

async function verifyPaths(items: VerifyItem[]) {
  const rows: Array<{ label: string; path: string; ok: boolean; kind: VerifyKind }> = [];
  for (const item of items) {
    const ok = await statExists(item.path, item.kind);
    rows.push({ label: item.label, path: item.path, ok, kind: item.kind });
  }
  const longest = rows.reduce((m, r) => Math.max(m, r.label.length), 0);
  let missing = 0;
  rows.forEach((r) => {
    const mark = r.ok ? chalk.green('✓') : chalk.red('✗');
    if (!r.ok) missing += 1;
    console.log(`${mark} ${r.label.padEnd(longest)}  ${r.path}`);
  });
  return missing;
}

export function registerVerifyCommands(program: Command) {
  program
    .command('verify')
    .description('Verifica estrutura de pastas/arquivos essenciais do MT5 (data_dir)')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const base = normalizePath(info.data_dir);
      const expected: VerifyItem[] = [
        { label: 'data_dir', path: base, kind: 'dir' },
        { label: 'Bases', path: path.join(base, 'Bases'), kind: 'dir' },
        { label: 'Config', path: path.join(base, 'Config'), kind: 'dir' },
        { label: 'Logs', path: path.join(base, 'Logs'), kind: 'dir' },
        { label: 'MQL5', path: path.join(base, 'MQL5'), kind: 'dir' },
        { label: 'Profiles', path: path.join(base, 'Profiles'), kind: 'dir' },
        { label: 'Templates', path: path.join(base, 'Templates'), kind: 'dir' },
        { label: 'Tester', path: path.join(base, 'Tester'), kind: 'dir' },
        { label: 'origin.txt', path: path.join(base, 'origin.txt'), kind: 'file' },
        // Config files
        { label: 'accounts.dat', path: path.join(base, 'Config', 'accounts.dat'), kind: 'file' },
        { label: 'common.ini', path: commonIniPath(base), kind: 'file' },
        { label: 'metaeditor.ini', path: path.join(base, 'Config', 'metaeditor.ini'), kind: 'file' },
        { label: 'terminal.ini', path: path.join(base, 'Config', 'terminal.ini'), kind: 'file' },
        { label: 'servers.dat', path: path.join(base, 'Config', 'servers.dat'), kind: 'file' },
        // MQL5 structure
        { label: 'Experts', path: path.join(base, 'MQL5', 'Experts'), kind: 'dir' },
        { label: 'Indicators', path: path.join(base, 'MQL5', 'Indicators'), kind: 'dir' },
        { label: 'Scripts', path: path.join(base, 'MQL5', 'Scripts'), kind: 'dir' },
        { label: 'Include', path: path.join(base, 'MQL5', 'Include'), kind: 'dir' },
        { label: 'Files', path: path.join(base, 'MQL5', 'Files'), kind: 'dir' },
        { label: 'Images', path: path.join(base, 'MQL5', 'Images'), kind: 'dir' },
        { label: 'Libraries', path: path.join(base, 'MQL5', 'Libraries'), kind: 'dir' },
        { label: 'Logs MQL5', path: path.join(base, 'MQL5', 'Logs'), kind: 'dir' },
        { label: 'Presets', path: path.join(base, 'MQL5', 'Presets'), kind: 'dir' },
        { label: 'Profiles/Charts', path: path.join(base, 'MQL5', 'Profiles', 'Charts'), kind: 'dir' },
        { label: 'Profiles/Templates', path: path.join(base, 'MQL5', 'Profiles', 'Templates'), kind: 'dir' },
        { label: 'Profiles/SymbolSets', path: path.join(base, 'MQL5', 'Profiles', 'SymbolSets'), kind: 'dir' },
        { label: 'Profiles/Tester', path: path.join(base, 'MQL5', 'Profiles', 'Tester'), kind: 'dir' },
        // Tester folders
        { label: 'Tester logs', path: path.join(base, 'Tester', 'logs'), kind: 'dir' },
        { label: 'Tester Cache', path: path.join(base, 'Tester', 'Cache'), kind: 'dir' },
      ];
      const missing = await verifyPaths(expected);
      if (missing === 0) {
        console.log(chalk.green('\nEstrutura OK.'));
      } else {
        console.log(chalk.yellow(`\nItens ausentes: ${missing}.`));
      }
    });
}
