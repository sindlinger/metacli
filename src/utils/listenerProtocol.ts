import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import crypto from 'crypto';
import { ProjectInfo } from '../config/projectStore.js';
import { normalizePath } from './paths.js';
import { isListenerRunning, restartListenerInstance } from '../commands/listener.js';

export interface ListenerCommandResult {
  id: string;
  status: 'OK' | 'ERROR';
  ok: boolean;
  message: string;
  data: string[];
}

export interface ListenerCommandOptions {
  timeoutMs?: number;
  pollIntervalMs?: number;
  ensureRunning?: boolean;
}

function generateCommandId(): string {
  const stamp = Date.now().toString(36);
  const random = crypto.randomBytes(3).toString('hex');
  return `${stamp}-${random}`;
}

export function formatCommandLine(id: string, type: string, params: Array<string | number | null | undefined>): string {
  const serialized = params.map((value) => {
    if (value === null || value === undefined) return '';
    return String(value);
  });
  return [id, type, ...serialized].join('|');
}

export async function sendListenerCommand(
  info: ProjectInfo,
  type: string,
  params: Array<string | number | null | undefined> = [],
  options: ListenerCommandOptions = {}
): Promise<ListenerCommandResult> {
  if (!info.data_dir) {
    throw new Error('Projeto sem data_dir configurado.');
  }
  const ensureRunning = options.ensureRunning !== false;
  if (ensureRunning && !(await isListenerRunning(info.terminal))) {
    console.log(chalk.gray(`[listener] terminal não detectado para ${info.project}. Reiniciando...`));
    await restartListenerInstance({ project: info.project, profile: info.defaults?.profile ?? undefined });
  }
  const cmdId = generateCommandId();
  const dataRoot = normalizePath(info.data_dir);
  const filesDir = path.join(dataRoot, 'MQL5', 'Files');
  await fs.ensureDir(filesDir);
  const cmdFile = path.join(filesDir, `cmd_${cmdId}.txt`);
  const respFile = path.join(filesDir, `resp_${cmdId}.txt`);
  await fs.remove(respFile).catch(() => {});
  const payload = formatCommandLine(cmdId, type, params);
  await fs.writeFile(cmdFile, `${payload}\n`, 'utf8');
  console.log(chalk.gray(`[listener] >> ${path.basename(cmdFile)} (${type})`));
  const timeoutMs = options.timeoutMs ?? 15000;
  const pollMs = options.pollIntervalMs ?? 200;
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await fs.pathExists(respFile)) {
      const raw = await fs.readFile(respFile, 'utf8');
      const lines = raw.replace(/\r/g, '').split('\n').map((line) => line.trim()).filter((line) => line.length > 0);
      const [statusLine = 'ERROR', messageLine = 'Resposta vazia'] = lines;
      const status = statusLine.toUpperCase() === 'OK' ? 'OK' : 'ERROR';
      const data = lines.slice(2);
      const result: ListenerCommandResult = {
        id: cmdId,
        status,
        ok: status === 'OK',
        message: messageLine || `(listener ${status.toLowerCase()})`,
        data,
      };
      await fs.remove(cmdFile).catch(() => {});
      await fs.remove(respFile).catch(() => {});
      const color = result.ok ? chalk.green : chalk.red;
      console.log(color(`[listener] << ${status} (${type}) - ${result.message}`));
      return result;
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
  throw new Error(`Timeout aguardando resposta do listener (${type}). Verifique se o EA está em execução.`);
}
