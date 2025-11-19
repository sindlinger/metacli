import fs from 'fs';

export function tailFile(filePath: string, lines = 40): string {
  if (!fs.existsSync(filePath)) {
    return '';
  }
  const data = fs.readFileSync(filePath, 'utf8');
  const parts = data.trimEnd().split(/\r?\n/);
  return parts.slice(-lines).join('\n');
}
