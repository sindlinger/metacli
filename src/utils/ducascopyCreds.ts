import path from 'path';
import { execa } from 'execa';

const SCRIPT_DIR = '/mnt/c/git/reg-account-dukascopy';
const SCRIPT = path.join(SCRIPT_DIR, 'cli-demo-account.mjs');

export interface DucasCreds {
  login: string;
  senha: string;
  expira?: string;
  restaDias?: number;
}

function parseCredentials(output: string): DucasCreds | null {
  const line = output.split('\n').find((l) => l.includes('CREDENCIAIS ENCONTRADAS'));
  if (!line) return null;
  const loginMatch = /login:\s*([\w-]+)/i.exec(line);
  const senhaMatch = /senha:\s*([\w-]+)/i.exec(line);
  const expiraMatch = /expira:\s*([\d-:TZ]+)/i.exec(line);
  const restaMatch = /resta:\s*([\d.,]+)/i.exec(line);
  return {
    login: loginMatch?.[1] || '',
    senha: senhaMatch?.[1] || '',
    expira: expiraMatch?.[1],
    restaDias: restaMatch ? parseFloat(restaMatch[1].replace(',', '.')) : undefined,
  };
}

async function runCliDemo(args: string[]): Promise<string> {
  const { stdout } = await execa('node', [SCRIPT, ...args], {
    cwd: SCRIPT_DIR,
    env: { ...process.env },
  });
  return stdout;
}

export async function ensureDucasCreds(thresholdDays = 1): Promise<DucasCreds | null> {
  const read = async () => parseCredentials(await runCliDemo(['mail', '--headless', '--quiet']));
  let creds = await read();
  if (!creds || !creds.login) {
    await runCliDemo(['--headless', '--quiet']);
    creds = await read();
  }
  if (!creds || !creds.login) return null;
  const resta = creds.restaDias ?? 0;
  if (resta <= thresholdDays) {
    await runCliDemo(['--headless', '--quiet']);
    creds = (await read()) || creds;
  }
  return creds;
}

