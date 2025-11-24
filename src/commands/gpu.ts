import { Command } from 'commander';
import path from 'path';
import fs from 'fs';
import fsExtra from 'fs-extra';
import chalk from 'chalk';
import { execa } from 'execa';
import { ProjectStore, repoRoot } from '../config/projectStore.js';
import { normalizePath, platformIsWindows, resolvePowerShell, toWinPath } from '../utils/paths.js';
import os from 'os';

const store = new ProjectStore();

const RELEASE_REQUIRED_FILES: Array<[string, string]> = [
  ['mt-bridge.dll', 'Bridge DLL'],
  ['core.dll', 'Core DLL'],
  ['tester.dll', 'Tester alias'],
  ['cudart64_13.dll', 'CUDA runtime'],
  ['cufft64_12.dll', 'CUDA FFT'],
  ['cublas64_13.dll', 'CUDA cuBLAS'],
  ['cublasLt64_13.dll', 'CUDA cuBLASLt'],
  ['nvrtc64_130_0.dll', 'CUDA NVRTC'],
  ['nvrtc-builtins64_130.dll', 'CUDA NVRTC builtins'],
  ['MSVCP140.dll', 'MSVC runtime (msvcp140)'],
  ['VCRUNTIME140.dll', 'MSVC runtime (vcruntime140)'],
  ['VCRUNTIME140_1.dll', 'MSVC runtime (vcruntime140_1)'],
  ['api-ms-win-crt-runtime-l1-1-0.dll', 'Universal CRT (runtime)'],
  ['api-ms-win-crt-heap-l1-1-0.dll', 'Universal CRT (heap)'],
  ['api-ms-win-crt-convert-l1-1-0.dll', 'Universal CRT (convert)'],
  ['api-ms-win-crt-stdio-l1-1-0.dll', 'Universal CRT (stdio)'],
  ['api-ms-win-crt-string-l1-1-0.dll', 'Universal CRT (string)'],
  ['api-ms-win-crt-math-l1-1-0.dll', 'Universal CRT (math)'],
];

const DEFAULT_RELEASE_ROOTS: Array<string | undefined> = [
  process.env.MTCLI_RELEASE_ROOT,
  process.env.GEN2_ROOT,
  'C:/mql5/Gen2Alglib/Gen2alglibfft',
  '/mnt/c/mql5/Gen2Alglib/Gen2alglibfft',
];

const GPU_PROJECTS_DIR = path.join(repoRoot(), 'GPU-dll_projects');

function collectPaths(value: string, previous: string[] = []) {
  previous.push(value);
  return previous;
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toISOString().replace('T', ' ').substring(0, 19);
}

function printTable(rows: string[][], headers: string[]) {
  const widths = headers.map((header, idx) => Math.max(header.length, ...rows.map((row) => (row[idx] || '').length)));
  const line = (values: string[]) => values.map((value, idx) => value.padEnd(widths[idx], ' ')).join('  ');
  console.log(line(headers));
  console.log(line(widths.map((w) => '-'.repeat(w))));
  for (const row of rows) console.log(line(row));
}

function detectReleaseRoot(): string {
  for (const candidate of DEFAULT_RELEASE_ROOTS) {
    if (!candidate) continue;
    const normalized = normalizePath(candidate);
    if (fs.existsSync(normalized)) {
      return normalized;
    }
  }
  return normalizePath(repoRoot());
}

function resolveReleaseDir(customRelease: string | undefined, config: string | undefined): string {
  if (customRelease) return normalizePath(customRelease);
  const cfg = config || 'Release';
  const base = detectReleaseRoot();
  return normalizePath(path.join(base, 'build-win', cfg));
}

async function ensureJunction(libsPath: string, releaseDir: string): Promise<{ state: string; note: string }> {
  const libsAbs = normalizePath(libsPath);
  const releaseAbs = normalizePath(releaseDir);
  if (!fs.existsSync(releaseAbs) || !fs.statSync(releaseAbs).isDirectory()) {
    return { state: 'ERROR', note: `Release inexistente: ${releaseAbs}` };
  }

  if (fs.existsSync(libsAbs)) {
    try {
      const libsReal = fs.realpathSync(libsAbs);
      const releaseReal = fs.realpathSync(releaseAbs);
      if (libsReal === releaseReal) return { state: 'OK', note: 'already linked' };
    } catch {
      // ignore comparison errors
    }
    return { state: 'SKIP', note: 'path exists (remova manualmente para recriar o link)' };
  }

  await fsExtra.ensureDir(path.dirname(libsAbs));
  const libsWin = toWinPath(libsAbs);
  const releaseWin = toWinPath(releaseAbs);
  const powershell = resolvePowerShell();
  const script = `$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "${libsWin}" -Target "${releaseWin}" | Out-Null`;
  try {
    const { stdout, stderr } = await execa(powershell, ['-NoProfile', '-Command', script]);
    if (stdout?.trim()) console.log(stdout.trim());
    if (stderr?.trim()) console.log(stderr.trim());
    return { state: 'LINKED', note: releaseWin };
  } catch (error) {
    const message =
      (error as { stderr?: string }).stderr?.trim() ||
      (error as { stdout?: string }).stdout?.trim() ||
      (error as Error).message ||
      'mklink falhou';
    return { state: 'ERROR', note: message };
  }
}

function findAgentLibraries(): string[] {
  const candidates = ['C:/mql5/Tester', '/mnt/c/mql5/Tester'];
  const results: string[] = [];
  for (const candidate of candidates) {
    const normalized = normalizePath(candidate);
    if (!fs.existsSync(normalized) || !fs.statSync(normalized).isDirectory()) continue;
    for (const entry of fs.readdirSync(normalized)) {
      if (!entry.toLowerCase().startsWith('agent')) continue;
      const libs = path.join(normalized, entry, 'MQL5', 'Libraries');
      results.push(libs);
    }
  }
  return Array.from(new Set(results.map((p) => normalizePath(p))));
}

function reportReleaseArtifacts(releaseDir: string) {
  if (!fs.existsSync(releaseDir) || !fs.statSync(releaseDir).isDirectory()) {
    console.log(`\nRelease artifacts: diretório inexistente (${releaseDir})`);
    return;
  }
  const rows = RELEASE_REQUIRED_FILES.map(([fileName, label]) => {
    const fullPath = path.join(releaseDir, fileName);
    if (fs.existsSync(fullPath)) {
      const stat = fs.statSync(fullPath);
      return [label, fileName, 'OK', formatTimestamp(stat.mtimeMs), toWinPath(fullPath)];
    }
    return [label, fileName, 'MISSING', '-', toWinPath(fullPath)];
  });
  console.log('\nRelease artifacts:');
  printTable(rows, ['Artifact', 'File', 'State', 'Modified', 'Path']);
}

async function listGpuProjects() {
  if (!(await fsExtra.pathExists(GPU_PROJECTS_DIR))) {
    console.log(chalk.yellow(`Nenhum projeto GPU: pasta ausente (${GPU_PROJECTS_DIR})`));
    return;
  }
  const entries = await fsExtra.readdir(GPU_PROJECTS_DIR, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory());
  if (dirs.length === 0) {
    console.log(chalk.yellow('Nenhum projeto GPU encontrado.'));
    return;
  }
  for (const dir of dirs) {
    const full = path.join(GPU_PROJECTS_DIR, dir.name);
    const files = (await fsExtra.readdir(full)).filter((f) => f.toLowerCase().endsWith('.dll'));
    const mark = files.length > 0 ? chalk.green('DLLs') : chalk.red('sem DLL');
    console.log(`${chalk.cyan(dir.name)} ${mark}`);
    if (files.length) {
      files.sort().forEach((f) => console.log(`  - ${f}`));
    }
  }
}

async function listRootBuilds() {
  const root = repoRoot();
  const entries = await fsExtra.readdir(root, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory() && e.name.toLowerCase().startsWith('build'));
  if (dirs.length === 0) {
    console.log(chalk.yellow('Nenhuma pasta de build* encontrada no diretório raiz.'));
    return;
  }
  for (const dir of dirs) {
    const full = path.join(root, dir.name);
    const files: string[] = [];
    const walkDepth1 = await fsExtra.readdir(full, { withFileTypes: true });
    for (const entry of walkDepth1) {
      const p = path.join(full, entry.name);
      if (entry.isFile() && entry.name.toLowerCase().endsWith('.dll')) {
        files.push(path.relative(root, p));
      } else if (entry.isDirectory()) {
        const inner = await fsExtra.readdir(p);
        inner
          .filter((f) => f.toLowerCase().endsWith('.dll'))
          .forEach((f) => files.push(path.relative(root, path.join(p, f))));
      }
    }
    const mark = files.length > 0 ? chalk.green('DLLs') : chalk.red('sem DLL');
    console.log(`${chalk.magenta(dir.name)} ${mark}`);
    files.sort().forEach((f) => console.log(`  - ${f}`));
  }
}

function guessVsVarsBat(): string | null {
  const candidates = [
    'C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat',
    'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat',
    'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat',
    'C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat',
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

async function runBuildScript(projectDir: string, script: string, opts: { config?: string; arch?: string }) {
  if (!platformIsWindows()) {
    throw new Error('gpu build requer Windows/WSL com interop habilitado.');
  }
  const fullScript = path.join(projectDir, script);
  if (!(await fsExtra.pathExists(fullScript))) {
    throw new Error(`Script de build não encontrado: ${fullScript}`);
  }
  const vcvars = guessVsVarsBat();
  const cfg = opts.config || 'Release';
  const arch = opts.arch || 'x64';
  const cmd = vcvars
    ? `call "${vcvars}" && cd /d "${projectDir}" && ${script} ${cfg}`
    : `cd /d "${projectDir}" && ${script} ${cfg}`;
  await execa('cmd.exe', ['/C', cmd], { stdio: 'inherit', windowsHide: true });
}

export function registerGpuCommands(program: Command) {
  const gpu = program.command('gpu').description('Build/link das DLLs GPU');

  gpu
    .command('build')
    .description('Fluxo de build das DLLs GPU (placeholder)')
    .action(() => {
      console.log('gpu build: conecte com seu pipeline de build (em desenvolvimento).');
    });

  gpu
    .command('link')
    .description('Cria junctions MQL5\\Libraries → build Release (bridge/core/tester em modo compartilhado)')
    .option('--release <path>', 'Diretório Release explícito (default: build-win/<config>)')
    .option('--config <name>', 'Subpasta do build-win (default: Release)', 'Release')
    .option('--libs <path>', 'Caminho para MQL5\\Libraries (pode repetir)', collectPaths, [] as string[])
    .option('--project <id>', 'Projeto salvo em mtcli_projects.json (LEGADO; evite, usa ativo)')
    .option('--agents', 'Inclui agentes do Tester (agent-XXXX) no link', false)
    .action(async (opts) => {
      if (!platformIsWindows()) {
        throw new Error('gpu link requer Windows ou WSL (mklink /J).');
      }

      const releaseDir = resolveReleaseDir(opts.release, opts.config);
      if (!fs.existsSync(releaseDir) || !fs.statSync(releaseDir).isDirectory()) {
        throw new Error(`Diretório Release não encontrado: ${releaseDir}`);
      }

      const targets: { id: string; libs: string }[] = [];
      const manualLibs: string[] = opts.libs ?? [];
      if (manualLibs.length > 0) {
        manualLibs.forEach((value, index) => targets.push({ id: `manual${index + 1}`, libs: value }));
      } else {
        const project = await store.useOrThrow(opts.project);
        if (!project.libs) {
          throw new Error(`Projeto "${project.project}" não possui caminho libs configurado.`);
        }
        targets.push({ id: project.project, libs: project.libs });
      }

      if (opts.agents) {
        const agents = findAgentLibraries();
        agents.forEach((libs, index) => targets.push({ id: `agent${index + 1}`, libs }));
      }

      if (targets.length === 0) {
        throw new Error('Nenhum destino para link foi resolvido.');
      }

      console.log(chalk.bold(`\n[link] Release: ${toWinPath(releaseDir)}`));
      const rows: string[][] = [];
      let errors = 0;
      for (const target of targets) {
        const normalizedLibs = normalizePath(target.libs);
        const result = await ensureJunction(normalizedLibs, releaseDir);
        if (result.state === 'ERROR') errors += 1;
        rows.push([target.id, toWinPath(normalizedLibs), result.state, result.note]);
      }
      console.log('\nLink state:');
      printTable(rows, ['Target', 'Libs', 'State', 'Note']);
      reportReleaseArtifacts(releaseDir);
      if (errors > 0) {
        throw new Error(`Alguns links falharam (${errors}). Veja a tabela acima.`);
      }
    });

  gpu
    .command('list')
    .description('Lista projetos GPU em GPU-dll_projects e suas DLLs')
    .action(async () => {
      await listGpuProjects();
      console.log('');
      await listRootBuilds();
    });

  gpu
    .command('build')
    .description('Build genérico de um projeto GPU (agnóstico), chamando scripts existentes')
    .option('--dir <path>', 'Pasta do projeto GPU (default: GPU-dll_projects/<nome>)')
    .option('--name <name>', 'Nome do projeto em GPU-dll_projects (se preferir indicar só o nome)')
    .option('--script <file>', 'Script de build (default: build_cmd.bat)', 'build_cmd.bat')
    .option('--config <cfg>', 'Config (Release/Debug)', 'Release')
    .option('--arch <arch>', 'Arquitetura (x64)', 'x64')
    .option('--copy-to-libs', 'Copia o output (Release) para MQL5/Libraries do projeto ativo', false)
    .action(async (opts) => {
      const projDir = opts.dir
        ? normalizePath(opts.dir)
        : path.join(GPU_PROJECTS_DIR, opts.name || '');
      if (!projDir || projDir.endsWith(path.sep)) {
        throw new Error('Informe --dir <path> ou --name <subpasta em GPU-dll_projects>.');
      }
      if (!(await fsExtra.pathExists(projDir))) {
        throw new Error(`Projeto GPU não encontrado: ${projDir}`);
      }
      await runBuildScript(projDir, opts.script, { config: opts.config, arch: opts.arch });

      if (opts.copyToLibs) {
        const info = await store.useOrThrow();
        if (!info.data_dir) throw new Error('data_dir do projeto ativo não configurado.');
        const releaseDir = path.join(projDir, 'build-win', opts.config);
        const libsDest = path.join(info.data_dir, 'MQL5', 'Libraries');
        await fsExtra.ensureDir(libsDest);
        const dlls = (await fsExtra.readdir(releaseDir)).filter((f) => f.toLowerCase().endsWith('.dll') || f.toLowerCase().endsWith('.exe'));
        for (const f of dlls) {
          await fsExtra.copy(path.join(releaseDir, f), path.join(libsDest, f), { overwrite: true });
          console.log(chalk.green(`[gpu build] copiado ${f} -> ${libsDest}`));
        }
      }
    });
}
