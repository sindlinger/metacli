import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, projectsFilePath } from '../config/projectStore.js';
import { runCommand } from '../utils/shell.js';

function iniSet(filePath: string, section: string, key: string, value: string) {
  let lines: string[] = [];
  if (fs.existsSync(filePath)) {
    lines = fs.readFileSync(filePath, 'utf8').replace(/\r/g, '').split('\n');
  }
  let secStart = -1;
  let secEnd = lines.length;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('[') && line.endsWith(']')) {
      const name = line.slice(1, -1);
      if (name.toLowerCase() === section.toLowerCase()) {
        secStart = i;
      } else if (secStart >= 0) {
        secEnd = i;
        break;
      }
    }
  }
  if (secStart === -1) {
    lines.push(`[${section}]`);
    lines.push(`${key}=${value}`);
  } else {
    let found = false;
    for (let i = secStart + 1; i < secEnd; i++) {
      const line = lines[i];
      const idx = line.indexOf('=');
      if (idx > -1) {
        const k = line.slice(0, idx).trim();
        if (k.toLowerCase() === key.toLowerCase()) {
          lines[i] = `${key}=${value}`;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      lines.splice(secEnd, 0, `${key}=${value}`);
    }
  }
  fs.writeFileSync(filePath, lines.join('\n'), 'utf8');
}

const store = new ProjectStore();

function buildArgs(opts: any) {
  const args: string[] = [];
  if (opts.config) args.push(`/config:${opts.config}`);
  if (opts.profile) args.push(`/profile:${opts.profile}`);
  if (opts.portable) args.push('/portable');
  if (opts.datapath) args.push(`/datapath:${opts.datapath}`);
  return args;
}

function renderConfigTemplate(opts: any) {
  const lines: string[] = [];
  // Common/login/profile
  lines.push('[Common]');
  lines.push(opts.login ? `Login=${opts.login}` : 'Login=');
  lines.push(opts.password ? `Password=${opts.password}` : 'Password=');
  lines.push(opts.server ? `Server=${opts.server}` : 'Server=');
  lines.push(opts.certPassword ? `CertPassword=${opts.certPassword}` : 'CertPassword=');
  if (typeof opts.proxyEnable !== 'undefined') lines.push(`ProxyEnable=${opts.proxyEnable ? 1 : 0}`);
  if (typeof opts.proxyType !== 'undefined') lines.push(`ProxyType=${opts.proxyType}`);
  if (typeof opts.proxyAddress !== 'undefined') lines.push(`ProxyAddress=${opts.proxyAddress}`);
  if (typeof opts.proxyLogin !== 'undefined') lines.push(`ProxyLogin=${opts.proxyLogin}`);
  if (typeof opts.proxyPassword !== 'undefined') lines.push(`ProxyPassword=${opts.proxyPassword}`);
  if (typeof opts.keepPrivate !== 'undefined') lines.push(`KeepPrivate=${opts.keepPrivate ? 1 : 0}`);
  if (typeof opts.newsEnable !== 'undefined') lines.push(`NewsEnable=${opts.newsEnable ? 1 : 0}`);
  if (typeof opts.certInstall !== 'undefined') lines.push(`CertInstall=${opts.certInstall ? 1 : 0}`);
  if (typeof opts.mql5Login !== 'undefined') lines.push(`MQL5Login=${opts.mql5Login}`);
  if (typeof opts.mql5Password !== 'undefined') lines.push(`MQL5Password=${opts.mql5Password}`);
  lines.push(opts.profile ? `ProfileLast=${opts.profile}` : 'ProfileLast=');
  lines.push(opts.profile ? `Profile=${opts.profile}` : 'Profile=');
  lines.push('');

  // Charts section
  const hasCharts = Boolean(
    typeof opts.chartsProfile !== 'undefined' ||
      typeof opts.maxBars !== 'undefined' ||
      typeof opts.printColor !== 'undefined' ||
      typeof opts.saveDeleted !== 'undefined'
  );
  if (hasCharts) {
    lines.push('[Charts]');
    lines.push(typeof opts.chartsProfile !== 'undefined' ? `ProfileLast=${opts.chartsProfile}` : 'ProfileLast=');
    if (typeof opts.maxBars !== 'undefined') lines.push(`MaxBars=${opts.maxBars}`);
    if (typeof opts.printColor !== 'undefined') lines.push(`PrintColor=${opts.printColor ? 1 : 0}`);
    if (typeof opts.saveDeleted !== 'undefined') lines.push(`SaveDeleted=${opts.saveDeleted ? 1 : 0}`);
    lines.push('');
  }

  // Experts section
  const hasExperts = Boolean(
    typeof opts.allowLiveTrading !== 'undefined' ||
      typeof opts.allowDll !== 'undefined' ||
      typeof opts.expertsEnabled !== 'undefined' ||
      typeof opts.expertsAccount !== 'undefined' ||
      typeof opts.expertsProfile !== 'undefined'
  );
  if (hasExperts) {
    lines.push('[Experts]');
    if (typeof opts.allowLiveTrading !== 'undefined') lines.push(`AllowLiveTrading=${opts.allowLiveTrading ? 1 : 0}`);
    if (typeof opts.allowDll !== 'undefined') lines.push(`AllowDllImport=${opts.allowDll ? 1 : 0}`);
    if (typeof opts.expertsEnabled !== 'undefined') lines.push(`Enabled=${opts.expertsEnabled ? 1 : 0}`);
    if (typeof opts.expertsAccount !== 'undefined') lines.push(`Account=${opts.expertsAccount ? 1 : 0}`);
    if (typeof opts.expertsProfile !== 'undefined') lines.push(`Profile=${opts.expertsProfile ? 1 : 0}`);
    lines.push('');
  }

  // Objects section
  const hasObjects = Boolean(
    typeof opts.showPropsOnCreate !== 'undefined' ||
      typeof opts.selectOneClick !== 'undefined' ||
      typeof opts.magnetSens !== 'undefined'
  );
  if (hasObjects) {
    lines.push('[Objects]');
    if (typeof opts.showPropsOnCreate !== 'undefined') lines.push(`ShowPropertiesOnCreate=${opts.showPropsOnCreate ? 1 : 0}`);
    if (typeof opts.selectOneClick !== 'undefined') lines.push(`SelectOneClick=${opts.selectOneClick ? 1 : 0}`);
    if (typeof opts.magnetSens !== 'undefined') lines.push(`MagnetSens=${opts.magnetSens}`);
    lines.push('');
  }

  // Email section
  const hasEmail = Boolean(
    typeof opts.emailEnable !== 'undefined' ||
      typeof opts.emailServer !== 'undefined' ||
      typeof opts.emailAuth !== 'undefined' ||
      typeof opts.emailLogin !== 'undefined' ||
      typeof opts.emailPassword !== 'undefined' ||
      typeof opts.emailFrom !== 'undefined' ||
      typeof opts.emailTo !== 'undefined'
  );
  if (hasEmail) {
    lines.push('[Email]');
    if (typeof opts.emailEnable !== 'undefined') lines.push(`Enable=${opts.emailEnable ? 1 : 0}`);
    if (typeof opts.emailServer !== 'undefined') lines.push(`Server=${opts.emailServer}`);
    if (typeof opts.emailAuth !== 'undefined') lines.push(`Auth=${opts.emailAuth}`);
    if (typeof opts.emailLogin !== 'undefined') lines.push(`Login=${opts.emailLogin}`);
    if (typeof opts.emailPassword !== 'undefined') lines.push(`Password=${opts.emailPassword}`);
    if (typeof opts.emailFrom !== 'undefined') lines.push(`From=${opts.emailFrom}`);
    if (typeof opts.emailTo !== 'undefined') lines.push(`To=${opts.emailTo}`);
    lines.push('');
  }

  // StartUp section (auto-open chart / attach EA/script)
  const hasStartUp = Boolean(
    opts.startExpert ||
    opts.startExpertParams ||
    opts.startSymbol ||
    opts.startPeriod ||
    opts.startTemplate ||
    opts.startScript ||
    opts.startScriptParams ||
    opts.startShutdown === true
  );
  if (hasStartUp) {
    lines.push('[StartUp]');
    lines.push(opts.startExpert ? `Expert=${opts.startExpert}` : 'Expert=');
    lines.push(opts.startExpertParams ? `ExpertParameters=${opts.startExpertParams}` : 'ExpertParameters=');
    lines.push(opts.startSymbol ? `Symbol=${opts.startSymbol}` : 'Symbol=');
    lines.push(opts.startPeriod ? `Period=${opts.startPeriod}` : 'Period=');
    lines.push(opts.startTemplate ? `Template=${opts.startTemplate}` : 'Template=');
    lines.push(opts.startScript ? `Script=${opts.startScript}` : 'Script=');
    lines.push(opts.startScriptParams ? `ScriptParameters=${opts.startScriptParams}` : 'ScriptParameters=');
    lines.push(typeof opts.startShutdown !== 'undefined' ? `ShutdownTerminal=${opts.startShutdown ? 1 : 0}` : 'ShutdownTerminal=');
    lines.push('');
  }

  // Tester section (optional)
  const hasTester = Boolean(
    opts.testerExpert ||
    opts.testerParams ||
    opts.testerSymbol ||
    opts.testerPeriod ||
    opts.testerLogin ||
    typeof opts.testerModel !== 'undefined' ||
    typeof opts.testerExecution !== 'undefined' ||
    typeof opts.testerOptimization !== 'undefined' ||
    typeof opts.testerCriterion !== 'undefined' ||
    opts.testerFrom ||
    opts.testerTo ||
    typeof opts.testerForwardMode !== 'undefined' ||
    opts.testerForwardDate ||
    opts.testerReport ||
    opts.testerReplaceReport === true ||
    opts.testerShutdown === true ||
    opts.testerDeposit ||
    opts.testerCurrency ||
    opts.testerLeverage ||
    opts.testerUseLocal === true ||
    opts.testerUseRemote === true ||
    opts.testerUseCloud === true ||
    opts.testerVisual === true ||
    opts.testerPort
  );
  if (hasTester) {
    lines.push('[Tester]');
    lines.push(opts.testerExpert ? `Expert=${opts.testerExpert}` : 'Expert=');
    lines.push(opts.testerParams ? `ExpertParameters=${opts.testerParams}` : 'ExpertParameters=');
    lines.push(opts.testerSymbol ? `Symbol=${opts.testerSymbol}` : 'Symbol=');
    lines.push(opts.testerPeriod ? `Period=${opts.testerPeriod}` : 'Period=');
    lines.push(typeof opts.testerLogin !== 'undefined' ? `Login=${opts.testerLogin}` : 'Login=');
    lines.push(typeof opts.testerModel !== 'undefined' ? `Model=${opts.testerModel}` : 'Model=');
    lines.push(typeof opts.testerExecution !== 'undefined' ? `ExecutionMode=${opts.testerExecution}` : 'ExecutionMode=');
    lines.push(typeof opts.testerOptimization !== 'undefined' ? `Optimization=${opts.testerOptimization}` : 'Optimization=');
    lines.push(typeof opts.testerCriterion !== 'undefined' ? `OptimizationCriterion=${opts.testerCriterion}` : 'OptimizationCriterion=');
    lines.push(opts.testerFrom ? `FromDate=${opts.testerFrom}` : 'FromDate=');
    lines.push(opts.testerTo ? `ToDate=${opts.testerTo}` : 'ToDate=');
    lines.push(typeof opts.testerForwardMode !== 'undefined' ? `ForwardMode=${opts.testerForwardMode}` : 'ForwardMode=');
    lines.push(opts.testerForwardDate ? `ForwardDate=${opts.testerForwardDate}` : 'ForwardDate=');
    lines.push(opts.testerReport ? `Report=${opts.testerReport}` : 'Report=');
    lines.push(typeof opts.testerReplaceReport !== 'undefined' ? `ReplaceReport=${opts.testerReplaceReport ? 1 : 0}` : 'ReplaceReport=');
    lines.push(typeof opts.testerShutdown !== 'undefined' ? `ShutdownTerminal=${opts.testerShutdown ? 1 : 0}` : 'ShutdownTerminal=');
    lines.push(opts.testerDeposit ? `Deposit=${opts.testerDeposit}` : 'Deposit=');
    lines.push(opts.testerCurrency ? `Currency=${opts.testerCurrency}` : 'Currency=');
    lines.push(opts.testerLeverage ? `Leverage=${opts.testerLeverage}` : 'Leverage=');
    lines.push(typeof opts.testerUseLocal !== 'undefined' ? `UseLocal=${opts.testerUseLocal ? 1 : 0}` : 'UseLocal=');
    lines.push(typeof opts.testerUseRemote !== 'undefined' ? `UseRemote=${opts.testerUseRemote ? 1 : 0}` : 'UseRemote=');
    lines.push(typeof opts.testerUseCloud !== 'undefined' ? `UseCloud=${opts.testerUseCloud ? 1 : 0}` : 'UseCloud=');
    lines.push(typeof opts.testerVisual !== 'undefined' ? `Visual=${opts.testerVisual ? 1 : 0}` : 'Visual=');
    lines.push(opts.testerPort ? `Port=${opts.testerPort}` : 'Port=');
    lines.push('');
  }

  return lines.join('\n');
}

function renderTesterConfig(opts: any) {
  const lines: string[] = [];
  lines.push('[Tester]');
  lines.push(opts.expert ? `Expert=${opts.expert}` : 'Expert=');
  lines.push(opts.params ? `ExpertParameters=${opts.params}` : 'ExpertParameters=');
  lines.push(opts.symbol ? `Symbol=${opts.symbol}` : 'Symbol=');
  lines.push(opts.period ? `Period=${opts.period}` : 'Period=');
  lines.push(typeof opts.login !== 'undefined' ? `Login=${opts.login}` : 'Login=');
  lines.push(typeof opts.model !== 'undefined' ? `Model=${opts.model}` : 'Model=0');
  lines.push(typeof opts.execution !== 'undefined' ? `ExecutionMode=${opts.execution}` : 'ExecutionMode=');
  lines.push(typeof opts.optimization !== 'undefined' ? `Optimization=${opts.optimization}` : 'Optimization=0');
  lines.push(typeof opts.criterion !== 'undefined' ? `OptimizationCriterion=${opts.criterion}` : 'OptimizationCriterion=');
  lines.push(opts.from ? `FromDate=${opts.from}` : 'FromDate=');
  lines.push(opts.to ? `ToDate=${opts.to}` : 'ToDate=');
  lines.push(typeof opts.forwardMode !== 'undefined' ? `ForwardMode=${opts.forwardMode}` : 'ForwardMode=0');
  lines.push(opts.forwardDate ? `ForwardDate=${opts.forwardDate}` : 'ForwardDate=');
  lines.push(opts.report ? `Report=${opts.report}` : 'Report=mtcli_report');
  lines.push(typeof opts.replaceReport !== 'undefined' ? `ReplaceReport=${opts.replaceReport ? 1 : 0}` : 'ReplaceReport=0');
  lines.push(typeof opts.shutdown !== 'undefined' ? `ShutdownTerminal=${opts.shutdown ? 1 : 0}` : 'ShutdownTerminal=0');
  lines.push(opts.deposit ? `Deposit=${opts.deposit}` : 'Deposit=10000');
  lines.push(opts.currency ? `Currency=${opts.currency}` : 'Currency=USD');
  lines.push(opts.leverage ? `Leverage=${opts.leverage}` : 'Leverage=1:100');
  lines.push(typeof opts.useLocal !== 'undefined' ? `UseLocal=${opts.useLocal ? 1 : 0}` : 'UseLocal=');
  lines.push(typeof opts.useRemote !== 'undefined' ? `UseRemote=${opts.useRemote ? 1 : 0}` : 'UseRemote=');
  lines.push(typeof opts.useCloud !== 'undefined' ? `UseCloud=${opts.useCloud ? 1 : 0}` : 'UseCloud=');
  lines.push(typeof opts.visual !== 'undefined' ? `Visual=${opts.visual ? 1 : 0}` : 'Visual=0');
  lines.push(opts.spread ? `Spread=${opts.spread}` : 'Spread=0');
  lines.push(opts.port ? `Port=${opts.port}` : 'Port=');
  lines.push('');
  lines.push('[TesterInputs]');
  lines.push('');
  return lines.join('\n');
}

type Scope = 'data' | 'config' | 'logs' | 'mql5' | 'tester' | 'root';

function resolveScopeBase(info: any, scope: Scope): string {
  if (!info.data_dir && scope !== 'root') {
    throw new Error('data_dir não configurado para o projeto.');
  }
  switch (scope) {
    case 'data':
      return info.data_dir!;
    case 'config':
      return path.join(info.data_dir!, 'config');
    case 'logs':
      return path.join(info.data_dir!, 'Logs');
    case 'mql5':
      return path.join(info.data_dir!, 'MQL5');
    case 'tester':
      return path.join(info.data_dir!, 'Tester');
    case 'root':
    default:
      if (!info.terminal) throw new Error('terminal64.exe não configurado.');
      return path.dirname(info.terminal);
  }
}

async function statExists(target: string, type: 'file' | 'dir' | 'any' = 'any') {
  if (!(await fs.pathExists(target))) return false;
  if (type === 'any') return true;
  const st = await fs.stat(target);
  return type === 'file' ? st.isFile() : st.isDirectory();
}

function commonIniPath(dataDir: string) {
  return path.join(dataDir, 'config', 'common.ini');
}

function testerIniPath(dataDir: string, file?: string) {
  return path.join(dataDir, file || 'tester.ini');
}

export function registerConfigCommands(program: Command) {
  const term = program.command('config').description('Configurações e operações diretas do MT5');

  term
    .command('config-set')
    .description('Altera um par key=value em config/common.ini do projeto')
    .requiredOption('--key <key>', 'Chave (ex.: EnableDlls)')
    .requiredOption('--value <value>', 'Valor')
    .option('--section <name>', 'Seção (default Common)', 'Common')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const iniPath = commonIniPath(info.data_dir);
      await fs.ensureDir(path.dirname(iniPath));
      iniSet(iniPath, opts.section, opts.key, opts.value);
      console.log(chalk.green(`[terminal] ${opts.section}.${opts.key}=${opts.value} em ${iniPath}`));
    });

  term
    .command('paths')
    .description('Mostra caminhos principais do terminal do projeto')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const rows: Array<{ label: string; value: string; exists: boolean }> = [];
      const push = async (label: string, value?: string | null) => {
        if (!value) return;
        const exists = await fs.pathExists(value);
        rows.push({ label, value, exists });
      };
      await push('terminal', info.terminal);
      await push('metaeditor', info.metaeditor);
      await push('root', info.terminal ? path.dirname(info.terminal) : null);
      await push('data_dir', info.data_dir);
      await push('config', info.data_dir ? path.join(info.data_dir, 'config') : null);
      await push('logs', info.data_dir ? path.join(info.data_dir, 'Logs') : null);
      await push('mql5', info.data_dir ? path.join(info.data_dir, 'MQL5') : null);
      await push('experts', info.data_dir ? path.join(info.data_dir, 'MQL5', 'Experts') : null);
      await push('indicators', info.data_dir ? path.join(info.data_dir, 'MQL5', 'Indicators') : null);
      await push('scripts', info.data_dir ? path.join(info.data_dir, 'MQL5', 'Scripts') : null);
      await push('libraries', info.data_dir ? path.join(info.data_dir, 'MQL5', 'Libraries') : null);
      await push('presets', info.data_dir ? path.join(info.data_dir, 'MQL5', 'Profiles', 'Tester') : null);
      await push('profiles', info.data_dir ? path.join(info.data_dir, 'Profiles') : null);
      await push('templates', info.data_dir ? path.join(info.data_dir, 'Profiles', 'Templates') : null);
      await push('tester', info.data_dir ? path.join(info.data_dir, 'Tester') : null);
      await push('tester logs', info.data_dir ? path.join(info.data_dir, 'Tester', 'logs') : null);
      await push('tester cache', info.data_dir ? path.join(info.data_dir, 'Tester', 'Cache') : null);

      const longest = rows.reduce((m, r) => Math.max(m, r.label.length), 0);
      rows.forEach((r) => {
        const mark = r.exists ? chalk.green('✓') : chalk.red('✗');
        console.log(`${mark} ${r.label.padEnd(longest)}  ${r.value}`);
      });
    });

  term
    .command('exists')
    .description('Verifica se um arquivo/pasta existe dentro do terminal do projeto')
    .requiredOption('--path <relOrAbs>', 'Caminho (relativo ao escopo escolhido)')
    .option('--scope <data|config|logs|mql5|tester|root>', 'Escopo base', 'data')
    .option('--type <file|dir|any>', 'Tipo esperado', 'any')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const base = resolveScopeBase(info, opts.scope as Scope);
      const target = path.isAbsolute(opts.path) ? opts.path : path.join(base, opts.path);
      const ok = await statExists(target, opts.type as any);
      console.log(`${ok ? chalk.green('EXISTS') : chalk.red('MISSING')} ${target}`);
    });

  term
    .command('ls')
    .description('Lista conteúdo de uma pasta do terminal (data/config/logs/mql5/tester/root)')
    .option('--path <relOrAbs>', 'Pasta (relativa ao escopo)', '.')
    .option('--scope <data|config|logs|mql5|tester|root>', 'Escopo base', 'data')
    .option('--max <n>', 'Máximo de entradas', '200')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      const base = resolveScopeBase(info, opts.scope as Scope);
      const target = path.isAbsolute(opts.path) ? opts.path : path.join(base, opts.path);
      if (!(await fs.pathExists(target))) {
        console.log(chalk.red(`Caminho não existe: ${target}`));
        return;
      }
      const entries = await fs.readdir(target, { withFileTypes: true });
      const max = Number(opts.max) || 200;
      entries.slice(0, max).forEach((e) => {
        const name = e.name;
        const mark = e.isDirectory() ? chalk.blue('[dir]') : e.isFile() ? '     ' : '[oth]';
        console.log(`${mark} ${name}`);
      });
      if (entries.length > max) {
        console.log(chalk.gray(`... (${entries.length - max} ocultos, use --max para ver mais)`));
      }
    });

  term
    .command('env')
    .description('Mostra variáveis de ambiente relevantes do mtcli')
    .action(() => {
      const envs = [
        'MTCLI_BASE_TERMINAL',
        'MTCLI_BASE_TERMINAL_DIR',
        'MTCLI_MT5_INSTALLER_URL',
        'MTCLI_PROJECTS',
        'MTCLI_DATA_DIR',
        'MTCLI_DEBUG',
      ];
      console.log(chalk.cyan('Variáveis de ambiente:'));
      envs.forEach((k) => console.log(`  ${k}=${process.env[k] ?? ''}`));
    });

  term
    .command('path')
    .description('Mostra o caminho do mtcli_projects.json')
    .action(() => {
      console.log(chalk.cyan(projectsFilePath()));
    });

  term
    .command('config-show')
    .description('Mostra config/common.ini do projeto')
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const iniPath = commonIniPath(info.data_dir);
      if (!fs.existsSync(iniPath)) {
        console.log(chalk.yellow(`common.ini não encontrado em ${iniPath}`));
        return;
      }
      const content = await fs.readFile(iniPath, 'utf8');
      console.log(content);
    });

  term
    .command('config-enable-dlls')
    .description('Habilita DLLs e EAs no common.ini e aplica /portable se precisar')
    .option('--project <id>')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.data_dir) throw new Error('data_dir não configurado.');
      const iniPath = commonIniPath(info.data_dir);
      await fs.ensureDir(path.dirname(iniPath));
      iniSet(iniPath, 'Experts', 'AllowDllImport', '1');
      iniSet(iniPath, 'Experts', 'Enabled', '1');
      console.log(chalk.green(`[terminal] DLLs habilitadas em ${iniPath}`));
    });

  term
    .command('config-template')
    .description('Gera um common.ini básico para usar com /config')
    .option('--out <file>', 'Destino', 'common.ini')
    .option('--login <id>')
    .option('--password <pass>')
    .option('--server <srv>')
    .option('--cert-password <pass>')
    .option('--profile <name>')
    // Common/network
    .option('--proxy-enable', 'ProxyEnable=1', false)
    .option('--proxy-type <0|1|2>', 'ProxyType (0 SOCKS4, 1 SOCKS5, 2 HTTP)')
    .option('--proxy-address <host:port>', 'ProxyAddress')
    .option('--proxy-login <user>', 'ProxyLogin')
    .option('--proxy-password <pass>', 'ProxyPassword')
    .option('--keep-private', 'KeepPrivate=1', false)
    .option('--news-enable', 'NewsEnable=1', false)
    .option('--cert-install', 'CertInstall=1', false)
    .option('--mql5-login <user>', 'MQL5Login')
    .option('--mql5-password <pass>', 'MQL5Password')
    // Charts
    .option('--charts-profile <name>', 'ProfileLast em [Charts]')
    .option('--max-bars <n>', 'MaxBars')
    .option('--print-color', 'PrintColor=1', false)
    .option('--save-deleted', 'SaveDeleted=1', false)
    // Experts
    .option('--allow-live-trading', 'AllowLiveTrading=1', false)
    .option('--allow-dll', 'AllowDllImport=1', false)
    .option('--experts-enabled', 'Experts Enabled=1', false)
    .option('--experts-account', 'Account=1 (desabilita EAs ao mudar login)', false)
    .option('--experts-profile', 'Profile=1 (desabilita EAs ao mudar profile)', false)
    // Objects
    .option('--show-props-on-create', 'ShowPropertiesOnCreate=1', false)
    .option('--select-one-click', 'SelectOneClick=1', false)
    .option('--magnet-sens <n>', 'MagnetSens')
    // Email
    .option('--email-enable', 'Email Enable=1', false)
    .option('--email-server <srv>', 'Email Server')
    .option('--email-auth <blob>', 'Email Auth')
    .option('--email-login <user>', 'Email Login')
    .option('--email-password <pass>', 'Email Password')
    .option('--email-from <name@addr>', 'Email From')
    .option('--email-to <name@addr>', 'Email To')
    // StartUp
    .option('--start-expert <path>', 'EA para abrir na inicialização (Examples\\MACD\\MACD Sample)')
    .option('--start-expert-params <file>', 'Arquivo .set em MQL5\\presets')
    .option('--start-symbol <symbol>', 'Símbolo do gráfico criado na inicialização')
    .option('--start-period <period>', 'Período (M1, H1, D1)')
    .option('--start-template <tpl>', 'Template a aplicar ao gráfico (Profiles\\Templates)')
    .option('--start-script <path>', 'Script para abrir na inicialização')
    .option('--start-script-params <file>', 'Arquivo .set do script (MQL5\\presets)')
    .option('--start-shutdown', 'Encerrar terminal ao fim do script (ShutdownTerminal=1)', false)
    // Common/network/charts/experts/objects/email
    .option('--proxy-enable', 'ProxyEnable=1', false)
    .option('--proxy-type <0|1|2>')
    .option('--proxy-address <host:port>')
    .option('--proxy-login <user>')
    .option('--proxy-password <pass>')
    .option('--keep-private', 'KeepPrivate=1', false)
    .option('--news-enable', 'NewsEnable=1', false)
    .option('--cert-install', 'CertInstall=1', false)
    .option('--mql5-login <user>')
    .option('--mql5-password <pass>')
    .option('--charts-profile <name>')
    .option('--max-bars <n>')
    .option('--print-color', 'PrintColor=1', false)
    .option('--save-deleted', 'SaveDeleted=1', false)
    .option('--allow-live-trading', 'AllowLiveTrading=1', false)
    .option('--allow-dll', 'AllowDllImport=1', false)
    .option('--experts-enabled', 'Experts Enabled=1', false)
    .option('--experts-account', 'Account=1', false)
    .option('--experts-profile', 'Profile=1', false)
    .option('--show-props-on-create', 'ShowPropertiesOnCreate=1', false)
    .option('--select-one-click', 'SelectOneClick=1', false)
    .option('--magnet-sens <n>')
    .option('--email-enable', 'Email Enable=1', false)
    .option('--email-server <srv>')
    .option('--email-auth <blob>')
    .option('--email-login <user>')
    .option('--email-password <pass>')
    .option('--email-from <name@addr>')
    .option('--email-to <name@addr>')
    // Tester
    .option('--tester-expert <path>', 'EA para testar')
    .option('--tester-params <file>', 'Parâmetros do EA (Profiles\\Tester\\*.set)')
    .option('--tester-symbol <symbol>')
    .option('--tester-period <period>')
    .option('--tester-login <login>')
    .option('--tester-model <n>', 'Modelagem (0=ticks,1=OHLC,2=open,3=math,4=real)')
    .option('--tester-execution <ms>', 'ExecutionMode (0 normal, -1 aleatório, >0 delay ms)')
    .option('--tester-optimization <n>', '0=off,1=slow,2=fast,3=all symbols')
    .option('--tester-criterion <n>', 'OptimizationCriterion (0..7)')
    .option('--tester-from <YYYY.MM.DD>')
    .option('--tester-to <YYYY.MM.DD>')
    .option('--tester-forward-mode <n>', '0 off,1 1/2,2 1/3,3 1/4,4 custom')
    .option('--tester-forward-date <YYYY.MM.DD>')
    .option('--tester-report <name>', 'Arquivo/slug do relatório')
    .option('--tester-replace-report', 'Sobrescreve report existente (ReplaceReport=1)', false)
    .option('--tester-shutdown', 'Encerra terminal ao fim do teste', false)
    .option('--tester-deposit <value>', 'Depósito inicial')
    .option('--tester-currency <ccy>', 'Moeda do depósito')
    .option('--tester-leverage <ratio>', 'Alavancagem ex.: 1:100')
    .option('--tester-use-local', 'Usar agentes locais (UseLocal=1)', false)
    .option('--tester-use-remote', 'Usar agentes remotos (UseRemote=1)', false)
    .option('--tester-use-cloud', 'Usar MQL5 Cloud (UseCloud=1)', false)
    .option('--tester-visual', 'Modo visual', false)
    .option('--tester-port <n>', 'Porta do agente local')
    .action(async (opts) => {
      const content = renderConfigTemplate(opts);
      const outPath = path.resolve(opts.out);
      await fs.writeFile(outPath, content, 'utf8');
      console.log(chalk.green(`[terminal] template gerado em ${outPath}`));
    });

  term
    .command('tester-template')
    .description('Gera ini para Strategy Tester (uso com /config)')
    .option('--out <file>', 'Destino', 'tester.ini')
    .requiredOption('--expert <name>', 'Experts\\EA.ex5')
    .requiredOption('--symbol <symbol>', 'Símbolo')
    .requiredOption('--period <period>', 'Período (H1, M15, etc.)')
    .option('--params <file>', 'ExpertParameters (.set em Profiles\\Tester)')
    .option('--login <login>', 'Login simulado')
    .option('--model <n>', 'Modelagem (0=tick)', '0')
    .option('--execution <ms>', 'ExecutionMode (0 normal, -1 aleatório, >0 delay ms)')
    .option('--optimization <n>', '0=off,1=slow,2=fast,3=all symbols', '0')
    .option('--criterion <n>', 'OptimizationCriterion (0..7)')
    .option('--from <YYYY.MM.DD>', 'Data inicial')
    .option('--to <YYYY.MM.DD>', 'Data final')
    .option('--forward-mode <n>', '0 off,1 1/2,2 1/3,3 1/4,4 custom', '0')
    .option('--forward-date <YYYY.MM.DD>', 'Data de início do forward')
    .option('--deposit <val>', 'Depósito', '10000')
    .option('--currency <ccy>', 'Moeda', 'USD')
    .option('--leverage <ratio>', 'Alavancagem ex.: 1:100', '1:100')
    .option('--use-local', 'UseLocal=1', false)
    .option('--use-remote', 'UseRemote=1', false)
    .option('--use-cloud', 'UseCloud=1', false)
    .option('--visual', 'Visual=1', false)
    .option('--spread <points>', 'Spread', '0')
    .option('--port <n>', 'Porta do agente local')
    .option('--report <name>', 'Nome base do report', 'mtcli_report')
    .option('--replace-report', 'Sobrescrever report existente', false)
    .option('--shutdown', 'Fechar terminal ao terminar o teste', false)
    .action(async (opts) => {
      const content = renderTesterConfig({
        expert: opts.expert,
        params: opts.params,
        symbol: opts.symbol,
        period: opts.period,
        login: opts.login,
        model: opts.model,
        execution: opts.execution,
        optimization: opts.optimization,
        criterion: opts.criterion,
        from: opts.from,
        to: opts.to,
        forwardMode: opts.forwardMode,
        forwardDate: opts.forwardDate,
        deposit: opts.deposit,
        currency: opts.currency,
        leverage: opts.leverage,
        useLocal: opts.useLocal,
        useRemote: opts.useRemote,
        useCloud: opts.useCloud,
        visual: opts.visual,
        spread: opts.spread,
        port: opts.port,
        report: opts.report,
        replaceReport: opts.replaceReport,
        shutdown: opts.shutdown,
      });
      const outPath = path.resolve(opts.out);
      await fs.writeFile(outPath, content, 'utf8');
      console.log(chalk.green(`[terminal] tester ini gerado em ${outPath}`));
    });

}
