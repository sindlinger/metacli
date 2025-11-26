import path from 'path';
import fs from 'fs-extra';
import { repoRoot, ProjectDefaults } from '../config/projectStore.js';
import { ensureDucasCreds } from './ducascopyCreds.js';

export async function deployFactoryTemplates(dataDir: string): Promise<void> {
  const factoryRoot = path.join(repoRoot(), 'factory', 'templates');
  if (!(await fs.pathExists(factoryRoot))) return;
  const destDir = path.join(dataDir, 'MQL5', 'Profiles', 'Templates');
  await fs.ensureDir(destDir);
  const files = await fs.readdir(factoryRoot);
   console.log(`[factory] templates: ${factoryRoot} -> ${destDir}`);
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
  const destDir = path.join(dataDir, 'Config'); // MetaTrader espera 'Config'
  await fs.ensureDir(destDir);
  const dest = path.join(destDir, 'common.ini');
  await fs.copy(src, dest, { overwrite: true });
  console.log(`[factory] config: ${src} -> ${dest}`);
}

export async function ensureAccountInIni(dataDir: string): Promise<void> {
  const iniPath = path.join(dataDir, 'Config', 'common.ini');
  if (!(await fs.pathExists(iniPath))) return;
  const creds = await ensureDucasCreds(1).catch((err) => {
    console.log(`[factory] aviso: não foi possível obter credenciais Ducas: ${err.message ?? err}`);
    return null;
  });
  if (!creds || !creds.login || !creds.senha) {
    console.log('[factory] aviso: credenciais Ducas não encontradas; common.ini permanece inalterado.');
    return;
  }
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
  console.log(`[factory] credenciais Ducas aplicadas no common.ini (login=${creds.login}, expira em ${creds.expira ?? 'desconhecido'})`);
}

function upsertKey(content: string, section: string, key: string, value: string): string {
  const sectionRe = new RegExp(`\\[${section}\\]([\\s\\S]*?)(?=\\n\\[|$)`, 'i');
  const match = content.match(sectionRe);
  const keyLine = `${key}=${value}`;
  if (match) {
    let body = match[1];
    const keyRe = new RegExp(`^${key}=.*$`, 'mi');
    if (keyRe.test(body)) {
      body = body.replace(keyRe, keyLine);
    } else {
      body = `${body.trimEnd()}\n${keyLine}\n`;
    }
    return content.replace(sectionRe, `[${section}]${body}`);
  }
  return `${content.trimEnd()}\n\n[${section}]\n${keyLine}\n`;
}

/**
 * Garante que o CommandListener seja carregado no start do MT5.
 * Usa defaults do projeto para Symbol/Period se existirem.
 */
export async function ensureCommandListenerStartup(
  dataDir: string,
  defaults?: ProjectDefaults
): Promise<void> {
  const iniPath = path.join(dataDir, 'Config', 'common.ini');
  if (!(await fs.pathExists(iniPath))) return;
  let content = await fs.readFile(iniPath, 'utf8');
  const symbol = defaults?.symbol || 'EURUSD';
  const period = defaults?.period || 'H1';

  content = upsertKey(content, 'StartUp', 'Expert', 'CommandListenerEA');
  content = upsertKey(content, 'StartUp', 'Symbol', symbol);
  content = upsertKey(content, 'StartUp', 'Period', period);

  await fs.writeFile(iniPath, content, 'utf8');
  console.log(`[factory] StartUp set: Expert=CommandListenerEA Symbol=${symbol} Period=${period} (${iniPath})`);
}
