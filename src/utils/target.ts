import path from 'path';
import { normalizePath } from './paths.js';
import { ProjectInfo } from '../config/projectStore.js';

function stripPrefix(name: string): string {
  const prefixes = ['i:', 'indi:', 'ind:', 'indicator:', 'e:', 'exp:', 'expert:'];
  for (const pre of prefixes) {
    if (name.toLowerCase().startsWith(pre)) {
      return name.slice(pre.length);
    }
  }
  return name;
}

function ensureExt(name: string): string {
  const lower = name.toLowerCase();
  if (lower.endsWith('.mq5') || lower.endsWith('.mqh') || lower.endsWith('.ex5')) return name;
  return `${name}.mq5`;
}

export type TargetKind = 'Indicators' | 'Experts';

function buildRelative(name: string, kind: TargetKind): string {
  const stripped = stripPrefix(name.trim());
  let rel = stripped.replace(/\//g, '\\');
  const kindLower = kind.toLowerCase();
  const relLower = rel.toLowerCase();
  if (relLower.startsWith(kindLower)) {
    // garante separador após o nome da pasta (caso o usuário tenha digitado "IndicatorsPOC...")
    const after = rel.slice(kind.length);
    if (!after.startsWith('\\') && after.length > 0) {
      rel = `${kind}\\${after.replace(/^\\+/, '')}`;
    }
  } else {
    rel = `${kind}\\${rel}`;
  }
  rel = ensureExt(rel);
  return rel;
}

export interface TargetResolution {
  file: string;
  kind: TargetKind;
  rel: string;        // relative to MQL5 root (with extension)
  relNoExt: string;   // relative without extension (MT5 attach typically usa sem .ex5)
  attachName: string; // basename without ext
}

/**
 * Resolve target file for indicator/expert with defaults and short names.
 */
export function resolveTarget(info: ProjectInfo, opts: { file?: string; indicator?: string; expert?: string }): TargetResolution {
  if (!info.data_dir) throw new Error('Projeto sem data_dir configurado.');
  const dataRoot = normalizePath(path.join(info.data_dir));

  // explicit file wins
  if (opts.file) {
    const file = normalizePath(opts.file);
    const lower = file.toLowerCase();
    const kind: TargetKind = lower.includes('indicator') ? 'Indicators' : 'Experts';
    const rel = path.relative(path.join(dataRoot, 'MQL5'), file).replace(/\//g, '\\');
    const relNoExt = rel.replace(/\.(mq5|mqh|ex5)$/i, '');
    return { file, kind, rel, relNoExt, attachName: path.basename(file, path.extname(file)) };
  }

  // indicator
  const ind = opts.indicator ?? info.defaults?.indicator ?? null;
  if (ind) {
    const rel = buildRelative(ind, 'Indicators');
    const file = normalizePath(path.join(dataRoot, 'MQL5', rel));
    const relNoExt = rel.replace(/\.(mq5|mqh|ex5)$/i, '');
    return { file, kind: 'Indicators', rel, relNoExt, attachName: path.basename(file, path.extname(file)) };
  }

  // expert
  const exp = opts.expert ?? info.defaults?.expert ?? null;
  if (exp) {
    const rel = buildRelative(exp, 'Experts');
    const file = normalizePath(path.join(dataRoot, 'MQL5', rel));
    const relNoExt = rel.replace(/\.(mq5|mqh|ex5)$/i, '');
    return { file, kind: 'Experts', rel, relNoExt, attachName: path.basename(file, path.extname(file)) };
  }

  throw new Error('Informe indicador ou expert (ou configure defaults no projeto).');
}
