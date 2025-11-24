import { Command } from 'commander';
import { execa } from 'execa';
import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';

const SCRIPT_DIR = '/mnt/c/git/reg-account-dukascopy';
const SCRIPT = path.join(SCRIPT_DIR, 'cli-demo-account.mjs');
const STATE_FILE = path.join(process.cwd(), 'ducascopy_state.json');

interface Creds {
  login: string;
  senha: string;
  enviado?: string;
  expira?: string;
  restaDias?: number;
}

function parseCredentials(output: string): Creds | null {
  // try condensed line first
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
    all: false,
  });
  return stdout;
}

async function saveState(data: any) {
  await fs.writeJson(STATE_FILE, data, { spaces: 2 });
}

async function loadState(): Promise<any> {
  if (await fs.pathExists(STATE_FILE)) {
    return fs.readJson(STATE_FILE).catch(() => ({}));
  }
  return {};
}

export function registerDucascopyCommands(program: Command) {
  const d = program.command('ducascopy').description('Gerencia conta demo Ducascopy via cli-demo-account');

  d.command('open-demo')
    .description('Verifica credencial demo; renova se faltar <=1 dia')
    .option('--threshold <dias>', 'Limite para renovar', '1')
    .action(async (opts) => {
      const threshold = parseFloat(opts.threshold);

      let out = await runCliDemo(['mail', '--headless', '--quiet']).catch(async (err) => {
        console.error(chalk.red('Falha ao consultar credenciais:'), err.shortMessage || err);
        return '';
      });
      let creds = parseCredentials(out || '');

      if (!creds || !creds.login) {
        console.log(chalk.yellow('Nenhuma credencial encontrada; criando nova...'));
        out = await runCliDemo(['--headless', '--quiet']);
        out += '\n' + (await runCliDemo(['mail', '--headless', '--quiet']));
        creds = parseCredentials(out || '');
      }

      if (!creds || !creds.login) {
        console.error(chalk.red('Não foi possível obter credenciais.'));
        return;
      }

      const resta = creds.restaDias ?? 0;
      const needRenew = resta <= threshold;

      if (needRenew) {
        console.log(chalk.yellow(`Faltam ${resta.toFixed(1)} dias — renovando demo...`));
        out = await runCliDemo(['--headless', '--quiet']);
        out += '\n' + (await runCliDemo(['mail', '--headless', '--quiet']));
        creds = parseCredentials(out || '') || creds;
      }

      await saveState({
        last_check: new Date().toISOString(),
        output: out,
        creds,
      });

      const ok = (creds.restaDias ?? 0) > threshold;
      const mark = ok ? chalk.green('OK') : chalk.red('EXPIRA EM <=1D');
      console.log(mark, `login=${creds.login} senha=${creds.senha} resta=${creds.restaDias ?? '?'} dias`);
      if (creds.expira) console.log(`expira: ${creds.expira}`);
    });

  d.command('mail')
    .description('Consulta credencial demo atual (sem renovar)')
    .action(async () => {
      const out = await runCliDemo(['mail', '--headless', '--quiet']);
      const creds = parseCredentials(out || '');
      if (!creds) {
        console.error(chalk.red('Nenhuma credencial encontrada.'));
        return;
      }
      console.log(out.trim());
      await saveState({ last_check: new Date().toISOString(), output: out, creds });
    });
}
