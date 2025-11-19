import path from 'path';
import os from 'os';

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
  return os.platform() === 'win32';
}
