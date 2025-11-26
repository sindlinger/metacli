import path from 'path';
import os from 'os';
import { execa } from 'execa';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { runCommand } from '../utils/shell.js';
import { toWslPath, toWinPath, resolvePowerShell } from '../utils/paths.js';

const store = new ProjectStore();

/**
 * Verifica se há um terminal rodando (best-effort).
 * Usa pgrep -f pelo nome do executável; em WSL pode não ver processos Windows.
 */
export async function isListenerRunning(exePath?: string): Promise<boolean> {
  if (!exePath) return false;
  const name = path.basename(exePath);

  // Caminho Windows equivalente (para checks via PowerShell)
  const winPath = toWinPath(exePath).toLowerCase();

  // Em Windows ou WSL, inspeciona processos via PowerShell (pega processos Windows).
  if (os.platform() === 'win32' || process.env.WSL_DISTRO_NAME || os.release().toLowerCase().includes('microsoft')) {
    try {
      const ps = resolvePowerShell();
      const script = `
$procs = Get-Process -Name terminal64 -ErrorAction SilentlyContinue | Where-Object {
  $_.Path -and $_.Path.ToLower() -eq "${winPath}"
}
if ($procs) { Write-Output "RUNNING" }`;
      const { stdout } = await execa(ps, ['-NoProfile', '-Command', script], { timeout: 4000 });
      return stdout.toString().toUpperCase().includes('RUNNING');
    } catch {
      // Se falhar, tenta pgrep abaixo.
    }
  }

  // Fallback: pgrep (Linux nativo)
  try {
    const { stdout } = await execa('pgrep', ['-f', name]);
    return stdout.trim().length > 0;
  } catch {
    return false;
  }
}

interface RestartOpts {
  project: string;
  profile?: string;
}

/**
 * Inicia o terminal do projeto em background para que o CommandListener EA responda.
 * Não bloqueia. Se o terminal não estiver configurado, lança erro.
 */
export async function restartListenerInstance(opts: RestartOpts): Promise<void> {
  const info = await store.useOrThrow(opts.project);
  if (!info.terminal) throw new Error('terminal64.exe não configurado.');
  const terminalExec = os.platform() === 'linux' ? toWslPath(info.terminal) : info.terminal;
  const args: string[] = [];
  if (opts.profile) args.push(`/profile:${opts.profile}`);
  if (info.defaults?.portable) args.push('/portable');
  if (info.data_dir) args.push(`/datapath:${info.data_dir}`);
  await runCommand(terminalExec, args, { stdio: 'ignore', detach: true });
  console.log(chalk.gray(`[listener] terminal iniciado em background (${terminalExec})`));
}

// Mantido por compatibilidade: alguns pontos usam stopListenerInstance.
export async function stopListenerInstance(): Promise<void> {
  // No-op placeholder; em versões futuras podemos localizar e encerrar o processo.
}

// Nenhuma função register* aqui; este arquivo só expõe helpers usados internamente.
