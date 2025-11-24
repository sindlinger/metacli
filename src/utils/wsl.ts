import { execa } from 'execa';

export async function toWindowsPath(p: string): Promise<string> {
  if (process.platform !== 'linux' || !process.env.WSL_DISTRO_NAME) return p;
  const { stdout } = await execa('wslpath', ['-w', p]);
  return stdout.trim();
}

