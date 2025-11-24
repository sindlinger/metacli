import path from 'path';
import { ProjectInfo } from '../config/projectStore.js';

export function resolveMetaeditorArgs(info: ProjectInfo, file: string, log?: string): string[] {
  const fileAbs = path.isAbsolute(file) ? file : path.resolve(file);
  const args: string[] = [`/compile:${fileAbs}`];
  if (log) {
    const logAbs = path.isAbsolute(log) ? log : path.resolve(log);
    args.push(`/log:${logAbs}`);
  }
  if (info.data_dir) {
    // ajuda o MetaEditor a achar includes/libs corretos
    args.push(`/include:${path.join(info.data_dir, 'MQL5', 'Include')}`);
    args.push(`/library:${path.join(info.data_dir, 'MQL5', 'Libraries')}`);
  }
  return args;
}
