import { execa } from 'execa';

export interface RunOptions {
  cwd?: string;
  detach?: boolean;
  stdio?: 'inherit' | 'ignore';
}

export async function runCommand(executable: string, args: string[], opts: RunOptions = {}) {
  const subprocess = execa(executable, args, {
    cwd: opts.cwd,
    stdio: opts.stdio || 'inherit',
    detached: opts.detach || false,
  });
  if (opts.detach) {
    subprocess.unref();
    return;
  }
  await subprocess;
}
