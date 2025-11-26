import { execa } from 'execa';

export async function toWindowsPath(p: string): Promise<string> {
  // se já for um caminho Windows (C:\ ou \\), devolve direto
  if (/^[A-Za-z]:\\/.test(p) || /^\\\\/.test(p)) return p;
  if (process.platform !== 'linux' || !process.env.WSL_DISTRO_NAME) return p;
  const { stdout } = await execa('wslpath', ['-w', p]);
  // corrige caracteres especiais (wslpath pode retornar substituições unicode em algumas fontes)
  return stdout.trim().replace(/\uF03A/g, ':').replace(/\uF05C/g, '\\');
}
