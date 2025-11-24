import path from 'path';
import fs from 'fs-extra';
import { repoRoot } from '../config/projectStore.js';
import { ensureDucasCreds } from './ducascopyCreds.js';

export async function deployFactoryTemplates(dataDir: string): Promise<void> {
  const factoryRoot = path.join(repoRoot(), 'factory', 'templates');
  if (!(await fs.pathExists(factoryRoot))) return;
  const destDir = path.join(dataDir, 'MQL5', 'Profiles', 'Templates');
  await fs.ensureDir(destDir);
  const files = await fs.readdir(factoryRoot);
  for (const f of files) {
    if (!f.toLowerCase().endsWith('.tpl')) continue;
    const src = path.join(factoryRoot, f);
    const dest = path.join(destDir, f);
    await fs.copy(src, dest, { overwrite: true });
    // se for mtcli-default, também sobrescreve Default.tpl para carregar ao abrir
    if (f.toLowerCase() === 'mtcli-default.tpl') {
      await fs.copy(src, path.join(destDir, 'Default.tpl'), { overwrite: true });
    }
  }
}

export async function deployFactoryConfig(dataDir: string): Promise<void> {
  const src = path.join(repoRoot(), 'factory', 'config', 'common.ini');
  if (!(await fs.pathExists(src))) return;
  const destDir = path.join(dataDir, 'config');
  await fs.ensureDir(destDir);
  const dest = path.join(destDir, 'common.ini');
  await fs.copy(src, dest, { overwrite: true });
}

export async function ensureAccountInIni(dataDir: string): Promise<void> {
  const iniPath = path.join(dataDir, 'config', 'common.ini');
  if (!(await fs.pathExists(iniPath))) return;
  const creds = await ensureDucasCreds(1).catch(() => null);
  if (!creds || !creds.login || !creds.senha) return;
  let content = await fs.readFile(iniPath, 'utf8');
  const setKey = (key: string, val: string) => {
    const re = new RegExp(`^${key}=.*$`, 'mi');
    if (re.test(content)) {
      content = content.replace(re, `${key}=${val}`);
    } else {
      content = content.replace(/\[Common\]/i, `[Common]\n${key}=${val}`);
    }
  };
  setKey('Login', creds.login);
  setKey('Password', creds.senha);
  await fs.writeFile(iniPath, content, 'utf8');
}
