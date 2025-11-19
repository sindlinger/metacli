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
  const match = /^([A-Za-z]):\\(.*)$/.exec(input);
  if (match) {
    const drive = match[1].toLowerCase();
    const rest = match[2].replace(/\\/g, '/');
    return `/mnt/${drive}/${rest}`;
  }
  return input;
}

export function normalizePath(input: string): string {
  if (!input) return input;
  const replaced = input.replace(/\\/g, path.sep);
  return path.resolve(replaced);
}

export function platformIsWindows(): boolean {
  return os.platform() === 'win32';
}
