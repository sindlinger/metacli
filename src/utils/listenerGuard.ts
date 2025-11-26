import fs from 'fs-extra';
import path from 'path';
import { ProjectInfo } from '../config/projectStore.js';
import { sendListenerCommand } from './listenerProtocol.js';
import { saveStatus } from './status.js';

/**
 * Verifica se o CommandListener responde ao PING sem reiniciar terminal.
 * Lança erro amigável se não responder.
 */
export async function ensureListenerAlive(info: ProjectInfo): Promise<void> {
  try {
    await sendListenerCommand(info, 'PING', [], { timeoutMs: 3000, ensureRunning: true, allowRestart: false });
  } catch (err) {
    const msg =
      'CommandListener não responde. Abra o terminal deste projeto ou rode mtcli init/reload (--restart) e garanta StartUp com CommandListener.';
    throw new Error(msg);
  }
}

async function writeHeartbeat(info: ProjectInfo) {
  if (!info.data_dir) return;
  const statusPath = path.join(info.data_dir, '.mtcli_status.json');
  const now = new Date().toISOString();
  const payload = { last_ping_ok: now };
  await fs.outputJson(statusPath, payload, { spaces: 2 });
  await saveStatus(info, { last_ping_ok: now });
}

export async function recordHealth(info: ProjectInfo): Promise<void> {
  await writeHeartbeat(info);
}

export async function ensureHealth(info: ProjectInfo, thresholdMs = 60000): Promise<void> {
  if (!info.data_dir) throw new Error('Projeto sem data_dir. Rode mtcli init.');
  const statusPath = path.join(info.data_dir, '.mtcli_status.json');
  if (!(await fs.pathExists(statusPath))) {
    throw new Error('Status ausente. Rode mtcli activate para registrar health.');
  }
  const data = await fs.readJson(statusPath).catch(() => null);
  const ts = data?.last_ping_ok ? Date.parse(data.last_ping_ok) : 0;
  if (!ts) {
    throw new Error('Status inválido. Rode mtcli activate para registrar health.');
  }
  const age = Date.now() - ts;
  if (age > thresholdMs) {
    throw new Error('Status expirado. Rode mtcli activate para atualizar health.');
  }
}
