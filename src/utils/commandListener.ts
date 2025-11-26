import fs from 'fs-extra';
import path from 'path';
import os from 'os';
import chalk from 'chalk';
import { execa } from 'execa';
import { resolveMetaeditorArgs } from './metaeditor.js';
import { ensureAccountInIni } from './factoryAssets.js';

function isWslEnv(): boolean {
  return (
    os.platform() === 'linux' &&
    (!!process.env.WSL_DISTRO_NAME || os.release().toLowerCase().includes('microsoft'))
  );
}
import { toWslPath } from './paths.js';

function escapeForPwshPath(p: string): string {
  return p.replace(/`/g, '``').replace(/"/g, '``"');
}

async function toWindowsPath(p: string): Promise<string> {
  if (os.platform() === 'win32') return p;
  if (isWslEnv()) {
    const { stdout } = await execa('wslpath', ['-w', p]);
    return stdout.trim();
  }
  return p;
}

function parseLoginPassword(cfgPath: string): { login?: string; password?: string; server?: string } {
  try {
    const content = fs.readFileSync(cfgPath, 'utf8');
    const login = /^Login=(.+)$/m.exec(content)?.[1]?.trim();
    const password = /^Password=(.+)$/m.exec(content)?.[1]?.trim();
    const server = /^Server=(.+)$/m.exec(content)?.[1]?.trim();
    return { login, password, server };
  } catch {
    return {};
  }
}

export async function killTerminalWindows(exePath: string): Promise<void> {
  // Mata terminal no Windows (ou WSL chamando taskkill)
  const isWin = os.platform() === 'win32';
  const isWsl = isWslEnv();
  if (!isWin && !isWsl) return;
  const cmd = isWin ? 'taskkill' : 'taskkill.exe';
  try {
    await execa(cmd, ['/F', '/IM', path.basename(exePath)], { stdio: 'ignore' });
    console.log(chalk.gray(`[cmdlistener] taskkill ${path.basename(exePath)}`));
  } catch {
    // ignore
  }
}

/**
 * Compila o CommandListenerEA.mq5 e garante que o .ex5 existe.
 * Lança erro se não conseguir compilar ou se o binário não aparecer.
 */
export async function ensureCommandListenerCompiled(dataDir: string): Promise<void> {
  const metaeditor = path.join(dataDir, 'MetaEditor64.exe');
  const src = path.join(dataDir, 'MQL5', 'Experts', 'CommandListenerEA.mq5');
  const dst = path.join(dataDir, 'MQL5', 'Experts', 'CommandListenerEA.ex5');
  const fallbackDst = path.join(dataDir, 'Experts', 'CommandListenerEA.ex5'); // às vezes o MetaEditor salva aqui
  if (!(await fs.pathExists(src))) {
    throw new Error(`CommandListenerEA.mq5 não encontrado em ${src}`);
  }
  const args = resolveMetaeditorArgs({ data_dir: dataDir, metaeditor } as any, src);
  const execPath = os.platform() === 'linux' ? toWslPath(metaeditor) : metaeditor;
  console.log(chalk.gray('[cmdlistener] compilando CommandListenerEA...'));
  const res = await execa(execPath, args, { stdio: 'inherit', windowsHide: false, reject: false });
  let ex5Exists = await fs.pathExists(dst);
  if (!ex5Exists && (await fs.pathExists(fallbackDst))) {
    // copia para o lugar correto
    await fs.copy(fallbackDst, dst, { overwrite: true });
    ex5Exists = true;
  }
  if (!ex5Exists) {
    throw new Error(`CommandListenerEA.ex5 não gerado (exitCode=${res.exitCode}).`);
  }
  if (res.exitCode !== 0) {
    console.log(chalk.yellow(`[cmdlistener] MetaEditor retornou exitCode=${res.exitCode}, mas .ex5 foi gerado.`));
  }
  console.log(chalk.green('[cmdlistener] CommandListenerEA compilado com sucesso.'));
}

/**
 * Tenta matar somente o terminal correspondente ao caminho informado.
 * No WSL/Linux usa pkill -f; se não existir ignora.
 */
export async function killTerminalIfRunning(exePath: string): Promise<void> {
  const isLinux = os.platform() === 'linux';
  if (isLinux) {
    try {
      await execa('pkill', ['-f', exePath]);
      console.log(chalk.gray(`[cmdlistener] matei terminal existente (pkill): ${exePath}`));
    } catch {
      // ignorar se não estava rodando
    }
  }
  await killTerminalWindows(exePath);
}

/**
 * Mata apenas o terminal cujo comando contém o datapath informado.
 */
export async function killTerminalByDatapath(datapath: string): Promise<void> {
  const isWin = os.platform() === 'win32';
  const isWsl = isWslEnv();
  if (!isWin && !isWsl) return;
  const dpWin = await toWindowsPath(datapath);
  const dpEsc = escapeForPwshPath(dpWin);
  const ps = isWin ? 'powershell.exe' : 'powershell.exe';
  const script = `
$procs = Get-WmiObject Win32_Process -Filter "Name='terminal64.exe'" | Where-Object {
  $_.CommandLine -and $_.CommandLine -match "${dpEsc}"
}
foreach ($p in $procs) {
  try { Stop-Process -Id $p.ProcessId -Force } catch {}
}
`;
  try {
    await execa(ps, ['-NoProfile', '-Command', script], { stdio: 'ignore' });
    console.log(chalk.gray(`[cmdlistener] taskkill por datapath (${datapath})`));
  } catch {
    // ignore
  }
}

/**
 * Inicia o terminal no Windows (ou WSL chamando powershell) sem bloquear.
 */
export async function startTerminalWindows(terminalPath: string, dataDir: string): Promise<void> {
  const isWin = os.platform() === 'win32';
  const isWsl = isWslEnv();
  if (!isWin && !isWsl) return;
  // garante credenciais atualizadas antes de subir
  await ensureAccountInIni(dataDir).catch(() => {});

  // garante profile mtcli-auto existindo (copia da Default)
  const profileName = 'mtcli-auto';
  const profilesDir = path.join(dataDir, 'Profiles');
  const defaultProfile = path.join(profilesDir, 'Default');
  const targetProfile = path.join(profilesDir, profileName);
  if (!(await fs.pathExists(targetProfile))) {
    if (await fs.pathExists(defaultProfile)) {
      await fs.copy(defaultProfile, targetProfile, { overwrite: false, errorOnExist: false });
    } else {
      await fs.ensureDir(targetProfile);
    }
  }
  const termWin = await toWindowsPath(terminalPath);
  const dpWin = await toWindowsPath(dataDir);
  const cfgWin = await toWindowsPath(path.join(dataDir, 'Config', 'common.ini'));
  const { login, password, server } = parseLoginPassword(cfgWin);
  const workWin = path.win32.dirname(termWin);
  const expertArg = '/expert:CommandListenerEA';
  const symbolArg = '/symbol:EURUSD';
  const periodArg = '/period:M1';
  const templateArg = '/template:mtcli-default.tpl';
  const configArg = `/config:${cfgWin}`;

  const ps = isWin
    ? 'powershell.exe'
    : '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe';

  // Gerar um script .ps1 no data_dir para evitar problemas de escaping em -Command
  const ps1Path = path.join(dataDir, 'run-terminal.ps1');
  const termEsc = escapeForPwshPath(termWin);
  const workEsc = escapeForPwshPath(workWin);
  const dpEsc = escapeForPwshPath(dpWin);
  const cfgEsc = escapeForPwshPath(cfgWin);
  const script = `
$argsList = @(
  '/portable',
  '/datapath:${dpEsc}',
  '/config:${cfgEsc}',
  '${expertArg}',
  '${symbolArg}',
  '${periodArg}',
  '${templateArg}',
  '/profile:${escapeForPwshPath(profileName)}'
  ${login && password ? `,'/login:${escapeForPwshPath(login)}','/password:${escapeForPwshPath(password)}'` : ''}
  ${server ? `,'/server:${escapeForPwshPath(server)}'` : ''}
  )
Start-Process -FilePath "${termEsc}" -WorkingDirectory "${workEsc}" -ArgumentList $argsList
`;
  await fs.writeFile(ps1Path, script, 'utf8');

  const ps1Win = await toWindowsPath(ps1Path);
  const res = await execa(ps, ['-NoProfile', '-File', ps1Win], { stdio: 'ignore', reject: false });
  if (res.exitCode !== 0) {
    console.log(chalk.yellow(`[cmdlistener] start (ps1) exitCode=${res.exitCode}`));
  }
  console.log(chalk.gray(`[cmdlistener] start terminal (ps1): ${termWin} /datapath:${dpWin}`));
}

/**
 * Verifica se um terminal específico está rodando (compara Path).
 */
export async function isTerminalRunning(terminalPath: string): Promise<boolean> {
  const isWin = os.platform() === 'win32';
  const isWsl = isWslEnv();
  if (!isWin && !isWsl) return false;
  const ps = isWin ? 'powershell.exe' : '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe';
  const termWin = (await toWindowsPath(terminalPath)).toLowerCase();
  const dpWin = (await toWindowsPath(path.dirname(terminalPath))).toLowerCase();
  const termEsc = escapeForPwshPath(termWin);
  const dpEsc = escapeForPwshPath(dpWin);
  const script = `
$procs = Get-WmiObject Win32_Process -Filter "Name='terminal64.exe'"
$hit = $procs | Where-Object {
  ($_.CommandLine -and $_.CommandLine.ToLower() -match "${dpEsc}") -or
  ($_.ExecutablePath -and $_.ExecutablePath.ToLower() -eq "${termEsc}")
}
if ($hit) { Write-Output "RUNNING" } else {
  $p2 = Get-Process -Name terminal64 -ErrorAction SilentlyContinue
  if ($p2) { Write-Output "RUNNING" } else { Write-Output "STOPPED" }
}
`;
  try {
    const { stdout } = await execa(ps, ['-NoProfile', '-Command', script], { stdio: 'pipe' });
    return stdout.toString().toLowerCase().includes('running');
  } catch {
    return false;
  }
}
