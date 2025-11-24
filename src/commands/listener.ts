import path from 'path';
import { execa } from 'execa';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { runCommand } from '../utils/shell.js';

const store = new ProjectStore();

/**
 * Verifica se há um terminal rodando (best-effort).
 * Usa pgrep -f pelo nome do executável; em WSL pode não ver processos Windows.
 */
export async function isListenerRunning(exePath?: string): Promise<boolean> {
  if (!exePath) return false;
  const name = path.basename(exePath);
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
  const args: string[] = [];
  if (opts.profile) args.push(`/profile:${opts.profile}`);
  if (info.defaults?.portable) args.push('/portable');
  if (info.data_dir) args.push(`/datapath:${info.data_dir}`);
  await runCommand(info.terminal, args, { stdio: 'ignore', detach: true });
  console.log(chalk.gray(`[listener] terminal iniciado em background (${info.terminal})`));
}

// Mantido por compatibilidade: alguns pontos usam stopListenerInstance.
export async function stopListenerInstance(): Promise<void> {
  // No-op placeholder; em versões futuras podemos localizar e encerrar o processo.
}

// Nenhuma função register* aqui; este arquivo só expõe helpers usados internamente.
