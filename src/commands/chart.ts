import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, ProjectInfo } from '../config/projectStore.js';
import { normalizePath, toWinPath, platformIsWindows, resolvePowerShell } from '../utils/paths.js';
import { restartListenerInstance, isListenerRunning } from './listener.js';
import { printLatestLogFromDataDir } from '../utils/logs.js';
import { execa } from 'execa';

const store = new ProjectStore();

function resolveSymbol(info: ProjectInfo, fallback?: string) {
  return fallback || (info.defaults?.symbol as string | undefined);
}

function resolvePeriod(info: ProjectInfo, fallback?: string) {
  return fallback || (info.defaults?.period as string | undefined);
}

function resolveSubwindow(info: ProjectInfo, fallback?: number) {
  const value = fallback ?? (info.defaults?.subwindow as number | undefined);
  return typeof value === 'number' && !Number.isNaN(value) ? value : 1;
}

function resolveIndicatorName(info: ProjectInfo, fallback?: string) {
  const value = fallback ?? (info.defaults?.indicator as string | undefined);
  if (!value) {
    throw new Error('Defina --indicator ou configure um padrão via `mtcli project defaults set --indicator <nome>`.');
  }
  return value;
}

function formatTimestamp(date: Date): string {
  const pad = (value: number) => value.toString().padStart(2, '0');
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(
    date.getMinutes()
  )}${pad(date.getSeconds())}`;
}

function sanitizeForFilename(input: string): string {
  return input.replace(/[^A-Za-z0-9._-]/g, '_');
}

function isAbsolutePathAnywhere(candidate: string): boolean {
  return path.isAbsolute(candidate) || path.win32.isAbsolute(candidate);
}

async function ensureListenerAndWrite(info: ProjectInfo, command: string, options: { delayMs?: number } = {}) {
  if (!(await isListenerRunning())) {
    await restartListenerInstance({ project: info.project, profile: info.defaults?.profile as string | undefined });
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  const dataDir = info.data_dir;
  if (!dataDir) {
    throw new Error('Projeto sem data_dir configurado.');
  }
  const dataRoot = normalizePath(dataDir);
  const fileDir = path.join(dataRoot, 'MQL5', 'Files');
  await fs.ensureDir(fileDir);
  const filePath = path.join(fileDir, 'cmd.txt');
  await fs.writeFile(filePath, command, 'utf8');
  console.log(chalk.green(`Comando gravado em ${filePath}`));
  if (options.delayMs && options.delayMs > 0) {
    await new Promise((resolve) => setTimeout(resolve, options.delayMs));
  }
  await printLatestLogFromDataDir(dataDir);
}

async function copyTemplate(dataDir: string, templatePath: string, target: 'chart' | 'tester'): Promise<string> {
  const src = normalizePath(templatePath);
  if (!(await fs.pathExists(src))) {
    throw new Error(`Template não encontrado: ${src}`);
  }
  const destRoot = normalizePath(dataDir);
  const destDir = path.join(
    destRoot,
    'MQL5',
    'Profiles',
    target === 'tester' ? 'Tester' : 'Templates'
  );
  await fs.ensureDir(destDir);
  const dest = path.join(destDir, path.basename(src));
  await fs.copyFile(src, dest);
  console.log(chalk.green(`[template] ${src} -> ${dest}`));
  return path.basename(dest);
}

async function ensureTemplateName(dataDir: string, options: { name?: string; file?: string }, target: 'chart' | 'tester' = 'chart'): Promise<string> {
  if (options.file) {
    return copyTemplate(dataDir, options.file, target);
  }
  if (options.name) {
    return options.name;
  }
  throw new Error('Informe --name ou --file (template).');
}

async function sendChartList(info: ProjectInfo) {
  await ensureListenerAndWrite(info, 'LIST_CHARTS', { delayMs: 800 });
}

async function sendCtrlSaveEnter(focusDelay = 250, confirmDelay = 250, overwriteDelay = 200) {
  if (!platformIsWindows()) {
    throw new Error('Automação via Ctrl+S requer Windows/WSL com terminal64.exe.');
  }
  const ps = resolvePowerShell();
  const script = `
$proc = Get-Process -Name terminal64 -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $proc) { throw "terminal64.exe não está em execução." }
Add-Type -AssemblyName Microsoft.VisualBasic
[Microsoft.VisualBasic.Interaction]::AppActivate($proc.Id) | Out-Null
Start-Sleep -Milliseconds ${Math.max(100, focusDelay)}
$wshell = New-Object -ComObject WScript.Shell
$wshell.SendKeys('^s')
Start-Sleep -Milliseconds ${Math.max(100, confirmDelay)}
$wshell.SendKeys('{ENTER}')
Start-Sleep -Milliseconds ${Math.max(100, overwriteDelay)}
$wshell.SendKeys('s')
`;
  await execa(ps, ['-NoProfile', '-Command', script], { stdio: 'inherit' });
}

export function registerChartCommands(program: Command) {
  const chart = program.command('chart').description('Opera gráficos e templates via listener');
  const screenshot = chart
    .command('screenshot')
    .description('Captura um screenshot do gráfico atual (MQL5/Files/screenshots por padrão)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .option('--output <path>', 'Arquivo PNG de saída (relativo ou absoluto)')
    .option('--width <px>', 'Largura em pixels', (val) => parseInt(val, 10))
    .option('--height <px>', 'Altura em pixels', (val) => parseInt(val, 10))
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const width = typeof opts.width === 'number' && Number.isFinite(opts.width) ? opts.width : 0;
      const height = typeof opts.height === 'number' && Number.isFinite(opts.height) ? opts.height : 0;
      const normalizedDataDir = normalizePath(info.data_dir);
      const filesDir = path.join(normalizedDataDir, 'MQL5', 'Files');
      const screenshotsDir = path.join(filesDir, 'screenshots');
      await fs.ensureDir(screenshotsDir);
      const providedOutput = opts.output as string | undefined;
      let desiredOutputPath: string;
      if (providedOutput) {
        if (isAbsolutePathAnywhere(providedOutput)) {
          desiredOutputPath = normalizePath(providedOutput);
        } else {
          desiredOutputPath = path.join(screenshotsDir, providedOutput);
        }
      } else {
        const filename = `${sanitizeForFilename(symbol)}-${period}-${formatTimestamp(new Date())}.png`;
        desiredOutputPath = path.join(screenshotsDir, filename);
      }
      const extension = path.extname(desiredOutputPath) || '.png';
      const rawBaseName = path.basename(desiredOutputPath, extension) || `${symbol}-${period}-${formatTimestamp(new Date())}`;
      const sanitizedBase = sanitizeForFilename(rawBaseName);
      const maxNameLen = Math.max(10, 63 - 'screenshots\\'.length - extension.length);
      const baseForChart = sanitizedBase.length > 0 ? sanitizedBase : `${symbol}-${period}`;
      const limitedBase = baseForChart.length > maxNameLen ? baseForChart.slice(-maxNameLen) : baseForChart;
      const chartRelativePath = path.join('screenshots', `${limitedBase}${extension}`);
      const chartRelativeWin = chartRelativePath.replace(/\//g, '\\');
      const tempShotPath = path.join(filesDir, chartRelativePath);
      const cmd = `SCREENSHOT;${symbol};${period};${chartRelativeWin};${width || 0};${height || 0}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 800 });
      const waitLimitMs = 4000;
      const start = Date.now();
      while (!(await fs.pathExists(tempShotPath)) && Date.now() - start < waitLimitMs) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      if (!(await fs.pathExists(tempShotPath))) {
        throw new Error(`Screenshot não encontrado em ${tempShotPath}. Verifique se o ChartScreenShot concluiu.`);
      }
      if (path.resolve(tempShotPath) !== path.resolve(desiredOutputPath)) {
        await fs.ensureDir(path.dirname(desiredOutputPath));
        await fs.copy(tempShotPath, desiredOutputPath);
        console.log(chalk.gray(`[screenshot] cópia para destino customizado: ${desiredOutputPath}`));
      }
      console.log(chalk.green(`[screenshot] arquivo (MQL5/Files): ${tempShotPath}`));
      console.log(chalk.green(`[screenshot] arquivo final: ${desiredOutputPath}`));
    });

  chart
    .command('screenshot-ind')
    .description('Captura um screenshot do gráfico que contém um indicador específico (ChartScreenShot)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--indicator <name>', 'Nome curto do indicador (ChartIndicatorName)')
    .option('--project <id>')
    .option('--output <path>', 'Arquivo PNG de saída (relativo ou absoluto)')
    .option('--width <px>', 'Largura em pixels', (val) => parseInt(val, 10))
    .option('--height <px>', 'Altura em pixels', (val) => parseInt(val, 10))
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const indicatorName = resolveIndicatorName(info, opts.indicator);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const width = typeof opts.width === 'number' && Number.isFinite(opts.width) ? opts.width : 0;
      const height = typeof opts.height === 'number' && Number.isFinite(opts.height) ? opts.height : 0;
      const normalizedDataDir = normalizePath(info.data_dir);
      const filesDir = path.join(normalizedDataDir, 'MQL5', 'Files');
      const screenshotsDir = path.join(filesDir, 'screenshots');
      await fs.ensureDir(screenshotsDir);
      const providedOutput = opts.output as string | undefined;
      let desiredOutputPath: string;
      if (providedOutput) {
        if (isAbsolutePathAnywhere(providedOutput)) {
          desiredOutputPath = normalizePath(providedOutput);
        } else {
          desiredOutputPath = path.join(screenshotsDir, providedOutput);
        }
      } else {
        const filename = `${sanitizeForFilename(symbol)}-${period}-${sanitizeForFilename(indicatorName)}-${formatTimestamp(
          new Date()
        )}.png`;
        desiredOutputPath = path.join(screenshotsDir, filename);
      }
      const extension = path.extname(desiredOutputPath) || '.png';
      const rawBaseName =
        path.basename(desiredOutputPath, extension) ||
        `${symbol}-${period}-${sanitizeForFilename(indicatorName)}-${formatTimestamp(new Date())}`;
      const sanitizedBase = sanitizeForFilename(rawBaseName);
      const maxNameLen = Math.max(10, 63 - 'screenshots\\'.length - extension.length);
      const baseForChart = sanitizedBase.length > 0 ? sanitizedBase : `${symbol}-${period}-${sanitizeForFilename(indicatorName)}`;
      const limitedBase = baseForChart.length > maxNameLen ? baseForChart.slice(-maxNameLen) : baseForChart;
      const chartRelativePath = path.join('screenshots', `${limitedBase}${extension}`);
      const chartRelativeWin = chartRelativePath.replace(/\//g, '\\');
      const tempShotPath = path.join(filesDir, chartRelativePath);
      const cmd = `SCREENSHOT_IND;${symbol};${period};${indicatorName};${chartRelativeWin};${width || 0};${height || 0}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 800 });
      const waitLimitMs = 4000;
      const start = Date.now();
      while (!(await fs.pathExists(tempShotPath)) && Date.now() - start < waitLimitMs) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      if (!(await fs.pathExists(tempShotPath))) {
        throw new Error(`Screenshot (por indicador) não encontrado em ${tempShotPath}. Verifique se o ChartScreenShot concluiu.`);
      }
      if (path.resolve(tempShotPath) !== path.resolve(desiredOutputPath)) {
        await fs.ensureDir(path.dirname(desiredOutputPath));
        await fs.copy(tempShotPath, desiredOutputPath);
        console.log(chalk.gray(`[screenshot-ind] cópia para destino customizado: ${desiredOutputPath}`));
      }
      console.log(chalk.green(`[screenshot-ind] arquivo (MQL5/Files): ${tempShotPath}`));
      console.log(chalk.green(`[screenshot-ind] arquivo final: ${desiredOutputPath}`));
    });

  screenshot
    .command('sweep')
    .description('Gera uma série de screenshots navegando pelo gráfico (ChartNavigate + ChartScreenShot)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .option('--steps <n>', 'Quantidade de capturas', (val) => parseInt(val, 10))
    .option('--shift <bars>', 'Quantidade de barras por passo (abs)', (val) => parseInt(val, 10))
    .option('--align <left|right>', 'Alinha a captura à esquerda ou direita (default: right)')
    .option('--width <px>', 'Largura em pixels', (val) => parseInt(val, 10))
    .option('--height <px>', 'Altura em pixels', (val) => parseInt(val, 10))
    .option('--format <png|gif|bmp>', 'Formato da imagem (default png)')
    .option('--delay <ms>', 'Atraso entre cada ChartScreenShot', (val) => parseInt(val, 10))
    .option('--output <dir>', 'Diretório destino para copiar a série (default: MQL5/Files/screenshots)')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const steps = Number.isFinite(opts.steps) ? Math.max(1, opts.steps) : 5;
      const shift = Number.isFinite(opts.shift) ? Math.max(1, Math.abs(opts.shift)) : 300;
      const align = (opts.align as string | undefined)?.toLowerCase() === 'left' ? 'LEFT' : 'RIGHT';
      const width = Number.isFinite(opts.width) ? opts.width : 0;
      const height = Number.isFinite(opts.height) ? opts.height : 0;
      const allowedFormats = new Set(['png', 'gif', 'bmp']);
      const fmt = ((opts.format as string | undefined)?.toLowerCase() || 'png').replace('.', '');
      if (!allowedFormats.has(fmt)) {
        throw new Error('Formato inválido. Use png, gif ou bmp.');
      }
      const extension = `.${fmt}`;
      const delay = Number.isFinite(opts.delay) ? Math.max(0, opts.delay) : 500;
      const normalizedDataDir = normalizePath(info.data_dir);
      const filesDir = path.join(normalizedDataDir, 'MQL5', 'Files');
      const screenshotsDir = path.join(filesDir, 'screenshots');
      await fs.ensureDir(screenshotsDir);
      const timestamp = formatTimestamp(new Date());
      const rawBase = `${sanitizeForFilename(symbol)}-${period}-${timestamp}-sw`;
      const sanitizedBase = sanitizeForFilename(rawBase);
      const numberingDigits = Math.max(3, String(steps).length);
      const folderPrefixLen = 'screenshots\\'.length;
      const maxNameLen = Math.max(8, 63 - folderPrefixLen - (1 + numberingDigits) - extension.length);
      const baseCandidate = sanitizedBase.length > 0 ? sanitizedBase : `${sanitizeForFilename(symbol)}-${period}`;
      const limitedBase = baseCandidate.length > maxNameLen ? baseCandidate.slice(-maxNameLen) : baseCandidate;
      const expectedFiles = [];
      for (let i = 1; i <= steps; i += 1) {
        const seq = String(i).padStart(numberingDigits, '0');
        const relativePosix = path.join('screenshots', `${limitedBase}-${seq}${extension}`);
        const relativeWin = relativePosix.replace(/\//g, '\\');
        expectedFiles.push({
          seq,
          relativePosix,
          relativeWin,
          absolute: path.join(filesDir, relativePosix),
        });
      }
      const cmd = `SCREENSHOT_SWEEP;${symbol};${period};screenshots;${limitedBase};${steps};${shift};${align};${width || 0};${height || 0};${fmt};${delay}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 800 });
      const waitLimitMs = Math.max(6000, steps * (delay + 500));
      const start = Date.now();
      while (true) {
        const missing = [];
        for (const file of expectedFiles) {
          if (!(await fs.pathExists(file.absolute))) {
            missing.push(file);
          }
        }
        if (missing.length === 0) {
          break;
        }
        if (Date.now() - start > waitLimitMs) {
          throw new Error(`Timeout aguardando screenshots: ${missing.map((f) => f.relativePosix).join(', ')}`);
        }
        await new Promise((resolve) => setTimeout(resolve, 200));
      }
      const outputDirParam = opts.output as string | undefined;
      let finalDir: string | undefined;
      if (outputDirParam) {
        finalDir = isAbsolutePathAnywhere(outputDirParam)
          ? normalizePath(outputDirParam)
          : path.resolve(process.cwd(), outputDirParam);
        await fs.ensureDir(finalDir);
        for (const file of expectedFiles) {
          const destination = path.join(finalDir, path.basename(file.absolute));
          await fs.copy(file.absolute, destination);
        }
      }
      console.log(chalk.green(`[screenshot sweep] arquivos (MQL5/Files):`));
      for (const file of expectedFiles) {
        console.log(`  - ${file.absolute}`);
      }
      if (finalDir) {
        console.log(chalk.green(`[screenshot sweep] cópias adicionais em: ${finalDir}`));
      }
    });

  chart
    .command('list')
    .description('Lista charts abertos e indicadores anexados via listener')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendChartList(info);
    });

  chart
    .command('close')
    .description('Fecha o gráfico (ChartClose) para o símbolo/período atual')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `CLOSE_CHART;${symbol};${period}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 400 });
    });

  chart
    .command('close-all')
    .description('Fecha todos os charts abertos (ChartFirst + ChartNext)')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await ensureListenerAndWrite(info, 'CLOSE_ALL', { delayMs: 500 });
    });

  chart
    .command('window-find')
    .description('Consulta ChartWindowFind para descobrir em qual subjanela está um indicador')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--indicator <name>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const indicatorName = resolveIndicatorName(info, opts.indicator);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `WINDOW_FIND;${symbol};${period};${indicatorName}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 400 });
    });

  chart
    .command('redraw')
    .description('Força um ChartRedraw() no gráfico atual ou em um chart_id específico')
    .option('--chart-id <id>', 'Chart ID alvo (default: 0 = atual)')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const chartId = opts.chartId ? String(opts.chartId) : '0';
      const cmd = `REDRAW_CHART;${chartId}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 400 });
    });

  chart
    .command('capture')
    .description('Anexa o indicador (opcional) e envia Ctrl+S / Enter no MT5 para salvar o gráfico pelo próprio terminal')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--indicator <name>', 'Nome do indicador (default usa projeto)')
    .option('--subwindow <index>', 'Subjanela do indicador', (val) => parseInt(val, 10))
    .option('--project <id>')
    .option('--skip-attach', 'Pula o ATTACH_IND antes de salvar', false)
    .option('--wait <ms>', 'Atraso em ms entre o attach e o Ctrl+S', (val) => parseInt(val, 10))
    .option('--focus-delay <ms>', 'Delay antes do Ctrl+S (focus)', (val) => parseInt(val, 10))
    .option('--confirm-delay <ms>', 'Delay entre Ctrl+S e Enter', (val) => parseInt(val, 10))
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      if (!opts.skipAttach) {
        const indicatorName = resolveIndicatorName(info, opts.indicator);
        const subwindow = resolveSubwindow(info, opts.subwindow);
        const attachCmd = `ATTACH_IND;${symbol};${period};${indicatorName};${subwindow}`;
        await ensureListenerAndWrite(info, attachCmd, { delayMs: 500 });
        const wait = Number.isFinite(opts.wait) ? Math.max(0, opts.wait) : 1500;
        if (wait > 0) {
          console.log(chalk.gray(`[capture] aguardando ${wait}ms para estabilizar o gráfico...`));
          await new Promise((resolve) => setTimeout(resolve, wait));
        }
      } else {
        console.log(chalk.gray('[capture] skip-attach habilitado; mantendo gráfico atual.'));
      }
      const focusDelay = Number.isFinite(opts.focusDelay) ? Math.max(50, opts.focusDelay) : 250;
      const confirmDelay = Number.isFinite(opts.confirmDelay) ? Math.max(50, opts.confirmDelay) : 250;
      await sendCtrlSaveEnter(focusDelay, confirmDelay);
      console.log(chalk.green('[capture] Ctrl+S + Enter + confirmação enviados ao MT5. Verifique MQL5\\Files pelo HTML/PNG gerado.'));
    });

  chart
    .command('drop-info')
    .description('Mostra ChartWindowOnDropped/ChartPriceOnDropped/etc do listener atual')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await ensureListenerAndWrite(info, 'DROP_INFO', { delayMs: 400 });
    });

  const template = chart.command('template').description('Gerencia templates (.tpl)');

  template
    .command('apply')
    .option('--name <tpl>', 'Nome do template já instalado (ex.: WaveSpecZZ.tpl)')
    .option('--file <path>', 'Arquivo .tpl a usar imediatamente')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const templateName = await ensureTemplateName(info.data_dir, { name: opts.name, file: opts.file }, 'chart');
      const cmd = `APPLY_TPL;${symbol};${period};${templateName}`;
      await ensureListenerAndWrite(info, cmd);
    });

  template
    .command('save')
    .option('--name <tpl>', 'Nome do template (adiciona .tpl se faltar)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const provided = opts.name as string | undefined;
      const defaultName = `${sanitizeForFilename(symbol)}-${period}-${formatTimestamp(new Date())}.tpl`;
      const candidate = provided ?? defaultName;
      const finalName = candidate.toLowerCase().endsWith('.tpl') ? candidate : `${candidate}.tpl`;
      const cmd = `SAVE_TPL;${symbol};${period};${finalName}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 400 });
    });
}

export function registerIndicatorCommands(program: Command) {
  const indicator = program.command('indicator').description('Gerencia indicadores via listener');

  indicator
    .command('add')
    .alias('attach')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--indicator <name>', 'Nome do indicador (usa default do projeto se omitido)')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10))
    .option('--project <id>', 'Projeto configurado')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir. Configure via mtcli project save --data-dir ...');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const indicatorName = resolveIndicatorName(info, opts.indicator);
      const subwindow = resolveSubwindow(info, opts.subwindow);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `ATTACH_IND;${symbol};${period};${indicatorName};${subwindow}`;
      await ensureListenerAndWrite(info, cmd);
    });

  indicator
    .command('del')
    .alias('detach')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--indicator <name>', 'Nome do indicador a remover (usa defaults se omitido)')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10))
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const indicatorName = resolveIndicatorName(info, opts.indicator);
      const subwindow = resolveSubwindow(info, opts.subwindow);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `DETACH_IND;${symbol};${period};${indicatorName};${subwindow}`;
      await ensureListenerAndWrite(info, cmd);
    });

  indicator
    .command('list')
    .description('Lista indicadores anexados nos charts (via log)')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      await sendChartList(info);
    });

  indicator
    .command('total')
    .description('Consulta ChartIndicatorsTotal para uma subjanela')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10))
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const subwindow = resolveSubwindow(info, opts.subwindow);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `IND_TOTAL;${symbol};${period};${subwindow}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 400 });
    });

  indicator
    .command('name')
    .description('Lê o nome curto de um indicador por índice (ChartIndicatorName)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10))
    .option('--index <idx>', 'Índice do indicador (default 0)', (val) => parseInt(val, 10))
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const subwindow = resolveSubwindow(info, opts.subwindow);
      const index = typeof opts.index === 'number' && Number.isFinite(opts.index) ? opts.index : 0;
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `IND_NAME;${symbol};${period};${subwindow};${index}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 400 });
    });

  indicator
    .command('handle')
    .alias('get')
    .description('Consulta ChartIndicatorGet para obter o handle de um indicador anexado')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--indicator <name>')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10))
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const indicatorName = resolveIndicatorName(info, opts.indicator);
      const subwindow = resolveSubwindow(info, opts.subwindow);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `IND_HANDLE;${symbol};${period};${subwindow};${indicatorName}`;
      await ensureListenerAndWrite(info, cmd, { delayMs: 400 });
    });

  indicator
    .command('redraw')
    .description('Força o indicador a reaplicar (detach/attach sequencial)')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--indicator <name>')
    .option('--subwindow <index>', 'Subjanela', (val) => parseInt(val, 10))
    .option('--project <id>')
    .option('--wait <ms>', 'Atraso entre o detach e o add', (val) => parseInt(val, 10))
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      const indicatorName = resolveIndicatorName(info, opts.indicator);
      const subwindow = resolveSubwindow(info, opts.subwindow);
      const wait = Number.isFinite(opts.wait) ? Math.max(0, opts.wait) : 600;
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const detach = `DETACH_IND;${symbol};${period};${indicatorName};${subwindow}`;
      await ensureListenerAndWrite(info, detach, { delayMs: 300 });
      if (wait > 0) {
        console.log(chalk.gray(`[indicator] aguardando ${wait}ms antes do reattach...`));
        await new Promise((resolve) => setTimeout(resolve, wait));
      }
      const attach = `ATTACH_IND;${symbol};${period};${indicatorName};${subwindow}`;
      await ensureListenerAndWrite(info, attach);
    });
}

export function registerExpertCommands(program: Command) {
  const expert = program.command('expert').description('Gerencia experts via listener');

  expert
    .command('add')
    .requiredOption('--expert <name>', 'Nome do EA (ex.: Examples\\MACD\\MACD Sample)')
    .option('--template <tpl>', 'Template (.tpl) já instalado com o EA')
    .option('--template-file <path>', 'Template (.tpl) a copiar e usar imediatamente')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const templateName = await ensureTemplateName(info.data_dir, { name: opts.template, file: opts.templateFile }, 'chart');
      const cmd = `ATTACH_EA;${symbol};${period};${opts.expert};${templateName}`;
      await ensureListenerAndWrite(info, cmd);
    });

  expert
    .command('del')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) {
        throw new Error('Projeto sem data_dir configurado.');
      }
      const symbol = resolveSymbol(info, opts.symbol);
      const period = resolvePeriod(info, opts.period);
      if (!symbol || !period) {
        throw new Error('Defina --symbol/--period ou configure defaults no projeto.');
      }
      const cmd = `DETACH_EA;${symbol};${period}`;
      await ensureListenerAndWrite(info, cmd);
    });
}
