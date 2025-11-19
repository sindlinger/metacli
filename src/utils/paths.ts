import path from 'path';
import os from 'os';
import fs from 'fs';

export function toWinPath(input: string): string {
  if (/^[A-Za-z]:\\/.test(input)) return input;
  if (input.startsWith('/mnt/')) {
    const segments = input.split('/');
    const driveLetter = (segments[2] || 'c').toLowerCase();
    const rest = segments.slice(3).join('/');
    const converted = rest.replace(/\//g, '\\');
    return `${driveLetter.toUpperCase()}:\\${converted}`;
  }
  return input.replace(/\//g, '\\');
}

export function toWslPath(input: string): string {
  if (input.startsWith('/mnt/')) return input;
  const normalized = input.replace(/\\/g, '/');
  const match = /^([A-Za-z]):\/(.*)$/.exec(normalized);
  if (match) {
    const drive = match[1].toLowerCase();
    const rest = match[2];
    return `/mnt/${drive}/${rest}`;
  }
  return input;
}

export function normalizePath(input: string): string {
  if (!input) return input;
  const wslLike = toWslPath(input);
  const replaced = wslLike.replace(/\\/g, path.sep);
  return path.resolve(replaced);
}

export function platformIsWindows(): boolean {
  if (os.platform() === 'win32') return true;
  if (os.platform() === 'linux') {
    const release = os.release().toLowerCase();
    if (release.includes('microsoft')) return true;
    if (process.env.WSL_DISTRO_NAME) return true;
  }
  return false;
}

const POWERSHELL_CANDIDATES = [
  process.env.POWERSHELL_EXE,
  'powershell.exe',
  '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
  '/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe',
];

export function resolvePowerShell(): string {
  const isPureWindows = os.platform() === 'win32';
  for (const candidate of POWERSHELL_CANDIDATES) {
    if (!candidate) continue;
    const hasPathSep = candidate.includes('/') || candidate.includes('\\');
    if (!hasPathSep) {
      if (isPureWindows) {
        return candidate;
      }
      continue;
    }
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error('powershell.exe não encontrado. Defina POWERSHELL_EXE ou ajuste os paths padrão.');
}
