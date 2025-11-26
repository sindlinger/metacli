import path from 'path';
import fs from 'fs-extra';
import { execa } from 'execa';
import { toWindowsPath } from './wsl.js';

interface InstallInfo {
  terminal: string;
  metaeditor: string;
  dataDir: string;
}

const LISTENER_SRC = path.join('mql5', 'Experts', 'CommandListener.mq5');

async function ensureCopied(dataDir: string): Promise<string> {
  const destDir = path.join(dataDir, 'MQL5', 'Experts');
  await fs.ensureDir(destDir);
  const src = path.join(process.cwd(), LISTENER_SRC);
  const dest = path.join(destDir, 'CommandListener.mq5');
  await fs.copy(src, dest, { overwrite: true });
  console.log(`[listener] source: ${src}`);
  console.log(`[listener] copied to: ${dest}`);
  return dest;
}

export async function deployCommandListener(install: InstallInfo): Promise<void> {
  const copied = await ensureCopied(install.dataDir);
  const metaWin = await toWindowsPath(install.metaeditor);
  const mq5Win = await toWindowsPath(copied);
  console.log(`[listener] compiling with MetaEditor: ${metaWin} /compile:${mq5Win}`);
  await execa('cmd.exe', ['/C', metaWin, `/compile:${mq5Win}`], { stdio: 'inherit', windowsHide: true });
  console.log('[listener] compile done');
}
