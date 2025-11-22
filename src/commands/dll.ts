import { Command } from 'commander';
import path from 'path';
import fs from 'fs';
import fsExtra from 'fs-extra';
import chalk from 'chalk';
import { execa } from 'execa';
import { ProjectStore, repoRoot } from '../config/projectStore.js';
import { normalizePath, platformIsWindows, resolvePowerShell, toWinPath } from '../utils/paths.js';
import { ensureTerminalStopped, restartListenerInstance } from './listener.js';

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

function collectPaths(value: string, previous: string[] = []) {
  previous.push(value);
  return previous;
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toISOString().replace('T', ' ').substring(0, 19);
}

function printTable(rows: string[][], headers: string[]) {
  const widths = headers.map((header, idx) => Math.max(header.length, ...rows.map((row) => (row[idx] || '').length)));
  const line = (values: string[]) =>
    values
      .map((value, idx) => value.padEnd(widths[idx], ' '))
      .join('  ');
  console.log(line(headers));
  console.log(line(widths.map((w) => '-'.repeat(w))));
  for (const row of rows) {
    console.log(line(row));
  }
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
  if (customRelease) {
    return normalizePath(customRelease);
  }
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
      if (libsReal === releaseReal) {
        return { state: 'OK', note: 'already linked' };
      }
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
    if (stdout?.trim()) {
      console.log(stdout.trim());
    }
    if (stderr?.trim()) {
      console.log(stderr.trim());
    }
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
    if (!fs.existsSync(normalized) || !fs.statSync(normalized).isDirectory()) {
      continue;
    }
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

export function registerDllCommands(program: Command) {
  const dll = program.command('dll').description('Builds e utilitários das DLLs');

  dll
    .command('build')
    .description('Fluxo de build das DLLs (placeholder)')
    .action(() => {
      console.log('dll build: conecte com seu pipeline de build (em desenvolvimento).');
    });

  dll
    .command('link')
    .description('Cria junctions MQL5\\Libraries → build Release (bridge/core/tester em modo compartilhado)')
    .option('--release <path>', 'Diretório Release explícito (default: build-win/<config>)')
    .option('--config <name>', 'Subpasta do build-win (default: Release)', 'Release')
    .option('--libs <path>', 'Caminho para MQL5\\Libraries (pode repetir)', collectPaths, [] as string[])
    .option('--project <id>', 'Projeto salvo em mtcli_projects.json')
    .option('--agents', 'Inclui agentes do Tester (agent-XXXX) no link', false)
    .action(async (opts) => {
      if (!platformIsWindows()) {
        throw new Error('dll link requer Windows ou WSL (mklink /J).');
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
        if (result.state === 'ERROR') {
          errors += 1;
        }
        rows.push([target.id, toWinPath(normalizedLibs), result.state, result.note]);
      }
      console.log('\nLink state:');
      printTable(rows, ['Target', 'Libraries', 'State', 'Note']);
      reportReleaseArtifacts(releaseDir);
      if (errors > 0) {
        throw new Error('Alguns links falharam. Revise a tabela acima.');
      }
    });

  dll
    .command('gpu-deploy')
    .description(
      'Copia a GpuEngine.dll compilada (temp/EngineDLL-GPU) para o MQL5\\\\Libraries do projeto, mantendo o build original livre.'
    )
    .option('--project <id>', 'Projeto salvo em mtcli_projects.json')
    .option('--source <path>', 'Caminho explícito para GpuEngine.dll (default: temp/EngineDLL-GPU/**/GpuEngine.dll)')
    .option('--restart', 'Reinicia o terminal/listener após a cópia', false)
    .action(async (opts) => {
      if (!platformIsWindows()) {
        throw new Error('dll gpu-deploy foi desenhado para Windows/WSL (terminal64.exe).');
      }

      const info = await store.useOrThrow(opts.project);
      if (!info.libs) {
        throw new Error(`Projeto "${info.project}" não possui libs configurado (MQL5\\Libraries).`);
      }

      // 1. Resolve origem da DLL (build canônico que fica sempre livre).
      const customSource = opts.source as string | undefined;
      let sourceDll: string | undefined;
      if (customSource) {
        const candidate = normalizePath(customSource);
        if (!fs.existsSync(candidate)) {
          throw new Error(`GpuEngine.dll não encontrada em --source: ${candidate}`);
        }
        sourceDll = candidate;
      } else {
        const candidates = [
          path.join(repoRoot(), 'temp', 'EngineDLL-GPU', 'runtime', 'bin', 'GpuEngine.dll'),
          path.join(repoRoot(), 'temp', 'EngineDLL-GPU', 'Dev', 'bin', 'GpuEngine.dll'),
        ];
        for (const candidate of candidates) {
          const normalized = normalizePath(candidate);
          if (fs.existsSync(normalized)) {
            sourceDll = normalized;
            break;
          }
        }
        if (!sourceDll) {
          throw new Error(
            'GpuEngine.dll não encontrada em temp/EngineDLL-GPU/runtime/bin nem em temp/EngineDLL-GPU/Dev/bin. Compile primeiro no EngineDLL-GPU.'
          );
        }
      }

      const libsDir = normalizePath(info.libs);
      await fsExtra.ensureDir(libsDir);
      const targetDll = normalizePath(path.join(libsDir, 'GpuEngine.dll'));

      console.log(chalk.bold('\n[gpu-deploy] Preparando ambiente'));
      console.log(`  Projeto : ${info.project}`);
      console.log(`  DataDir : ${info.data_dir}`);
      console.log(`  Libs    : ${toWinPath(libsDir)}`);
      console.log(`  Origem  : ${toWinPath(sourceDll)}`);

      // 2. Garante que o terminal não está com a DLL carregada.
      await ensureTerminalStopped();

      // 3. Se já existir uma GpuEngine.dll no MQL5\\Libraries, cria backup versionado.
      if (fs.existsSync(targetDll)) {
        const stat = fs.statSync(targetDll);
        const date = new Date(stat.mtimeMs || Date.now());
        const pad = (n: number) => n.toString().padStart(2, '0');
        const ts = `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(
          date.getHours()
        )}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
        const backupName = `GpuEngine_${ts}.dll`;
        const backupPath = normalizePath(path.join(libsDir, backupName));
        await fsExtra.copy(targetDll, backupPath);
        console.log(chalk.gray(`[gpu-deploy] Backup criado: ${toWinPath(backupPath)}`));
      }

      // 4. Copia a DLL canônica para o MQL5\\Libraries.
      await fsExtra.copy(sourceDll, targetDll);
      console.log(chalk.green(`[gpu-deploy] GpuEngine.dll atualizada em ${toWinPath(targetDll)}`));

      // 5. Opcionalmente reinicia o terminal com o listener.
      if (opts.restart) {
        console.log(chalk.gray('[gpu-deploy] Reiniciando terminal/listener após deploy...'));
        await restartListenerInstance({ project: info.project, profile: info.defaults?.profile as string | undefined });
      } else {
        console.log(
          chalk.gray('[gpu-deploy] Deploy concluído. Use `mtcli listener restart` se quiser relançar o terminal agora.')
        );
      }
    });
}
