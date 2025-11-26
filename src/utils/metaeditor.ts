import path from 'path';
import { ProjectInfo } from '../config/projectStore.js';
import { toWinPath } from './paths.js';

// Sempre devolve caminhos em formato Windows, pois o MetaEditor é Windows-only.
export function resolveMetaeditorArgs(info: ProjectInfo, file: string, log?: string): string[] {
  const fileAbs = path.isAbsolute(file) ? file : path.resolve(file);
  const args: string[] = [`/compile:${toWinPath(fileAbs)}`];
  // Força usar a pasta de dados local (portable) quando configurado no projeto.
  if (info.defaults?.portable) {
    args.push('/portable');
  }
  if (log) {
    const logAbs = path.isAbsolute(log) ? log : path.resolve(log);
    args.push(`/log:${toWinPath(logAbs)}`);
  }
  if (info.data_dir) {
    // Aponta para a raiz MQL5; o MetaEditor acrescenta \Include internamente
    args.push(`/include:${toWinPath(path.join(info.data_dir, 'MQL5'))}`);
    args.push(`/library:${toWinPath(path.join(info.data_dir, 'MQL5', 'Libraries'))}`);
  }
  return args;
}
