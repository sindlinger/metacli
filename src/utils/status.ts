import path from 'path';
import fs from 'fs-extra';
import { ProjectInfo } from '../config/projectStore.js';

export interface StatusData {
  last_ping_ok?: string;
  current_indicator?: string;
  current_symbol?: string;
  current_period?: string;
}

function statusPath(info: ProjectInfo): string {
  if (!info.data_dir) throw new Error('Projeto sem data_dir configurado.');
  return path.join(info.data_dir, '.mtcli_status.json');
}

export async function loadStatus(info: ProjectInfo): Promise<StatusData> {
  const file = statusPath(info);
  if (!(await fs.pathExists(file))) return {};
  return (await fs.readJson(file).catch(() => ({}))) as StatusData;
}

// Versão síncrona para renderização de help/status
export function loadStatusSync(info: ProjectInfo): StatusData {
  try {
    const file = statusPath(info);
    if (!fs.existsSync(file)) return {};
    const raw = fs.readFileSync(file, 'utf8');
    if (!raw.trim()) return {};
    return JSON.parse(raw) as StatusData;
  } catch {
    return {};
  }
}

export async function saveStatus(info: ProjectInfo, patch: Partial<StatusData>): Promise<void> {
  const file = statusPath(info);
  const current = (await loadStatus(info)) || {};
  const next: StatusData = { ...current, ...patch };
  await fs.outputJson(file, next, { spaces: 2 });
}
