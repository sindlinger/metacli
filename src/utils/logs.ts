import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import { normalizePath } from './paths.js';

export function tailFile(filePath: string, lines = 40): string {
  if (!fs.existsSync(filePath)) return '';
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

function latestLogFile(logDir: string): { file?: string; content?: string } {
  if (!fs.existsSync(logDir)) return {};
  const files = fs
    .readdirSync(logDir)
    .filter((name) => name.endsWith('.log'))
    .sort((a, b) => fs.statSync(path.join(logDir, b)).mtimeMs - fs.statSync(path.join(logDir, a)).mtimeMs);
  if (!files.length) return {};
  const latest = path.join(logDir, files[0]);
  return { file: latest, content: fs.readFileSync(latest, 'utf8') };
}

function extractAuthLines(content: string, max = 120): string[] {
  if (!content) return [];
  const lines = content.trimEnd().split(/\r?\n/);
  const authRe =
    /(auth|login|senha|password|account|invalid|failed|authorization|no connection|server|trade context)/i;
  const filtered = lines.filter((l) => authRe.test(l));
  return filtered.slice(-max);
}

/**
 * Coleta mensagens de autenticação/conexão nos logs do terminal e dos experts.
 */
export function collectAuthHints(dataDir?: string, limit = 120): string[] {
  if (!dataDir) return [];
  const root = normalizePath(dataDir);
  const termLogDir = path.join(root, 'Logs');
  const eaLogDir = path.join(root, 'MQL5', 'Logs');
  const hints: string[] = [];

  const { file: termFile, content: termContent } = latestLogFile(termLogDir);
  if (termContent) {
    const lines = extractAuthLines(termContent, limit);
    if (lines.length) {
      hints.push(`Terminal log (${path.basename(termFile as string)}):`);
      hints.push(...lines.slice(-limit));
    }
  }

  const { file: eaFile, content: eaContent } = latestLogFile(eaLogDir);
  if (eaContent) {
    const lines = extractAuthLines(eaContent, limit);
    if (lines.length) {
      hints.push(`MQL5 log (${path.basename(eaFile as string)}):`);
      hints.push(...lines.slice(-limit));
    }
  }

  return hints.slice(-limit);
}
