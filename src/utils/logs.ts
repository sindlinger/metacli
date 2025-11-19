import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import { normalizePath } from './paths.js';

export function tailFile(filePath: string, lines = 40): string {
  if (!fs.existsSync(filePath)) {
    return '';
  }
  const data = fs.readFileSync(filePath, 'utf8');
  const parts = data.trimEnd().split(/\r?\n/);
  return parts.slice(-lines).join('\n');
}

export function printLatestLogFromDataDir(dataDir?: string, limit = 40) {
  if (!dataDir) {
    console.log('Sem data_dir configurado para leitura de logs.');
    return;
  }
  const root = normalizePath(dataDir);
  const logDir = path.join(root, 'MQL5', 'Logs');
  if (!fs.existsSync(logDir)) {
    console.log(`Sem logs em ${logDir}`);
    return;
  }
  const files = fs
    .readdirSync(logDir)
    .filter((name) => name.endsWith('.log'))
    .sort((a, b) => fs.statSync(path.join(logDir, b)).mtimeMs - fs.statSync(path.join(logDir, a)).mtimeMs);
  if (files.length === 0) {
    console.log(`Nenhum log encontrado em ${logDir}`);
    return;
  }
  const latest = path.join(logDir, files[0]);
  console.log(chalk.bold(`Log recente (${files[0]}):`));
  console.log(tailFile(latest, limit));
}
