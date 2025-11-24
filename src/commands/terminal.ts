import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore } from '../config/projectStore.js';
import { runCommand } from '../utils/shell.js';

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
  return [
    '[Common]',
    opts.login ? `Login=${opts.login}` : 'Login=',
    opts.password ? `Password=${opts.password}` : 'Password=',
    opts.server ? `Server=${opts.server}` : 'Server=',
    opts.certPassword ? `CertPassword=${opts.certPassword}` : 'CertPassword=',
    opts.profile ? `ProfileLast=${opts.profile}` : 'ProfileLast=',
    opts.profile ? `Profile=${opts.profile}` : 'Profile=',
    opts.expert ? `Expert=${opts.expert}` : 'Expert=',
    opts.symbol ? `Symbol=${opts.symbol}` : 'Symbol=',
    opts.period ? `Period=${opts.period}` : 'Period=',
    opts.template ? `Template=${opts.template}` : 'Template=',
    '',
  ].join('\n');
}

function renderTesterConfig(opts: any) {
  return [
    '[Tester]',
    opts.expert ? `Expert=${opts.expert}` : 'Expert=',
    opts.symbol ? `Symbol=${opts.symbol}` : 'Symbol=',
    opts.period ? `Period=${opts.period}` : 'Period=',
    opts.model ? `Model=${opts.model}` : 'Model=0',
    opts.spread ? `Spread=${opts.spread}` : 'Spread=0',
    opts.deposit ? `Deposit=${opts.deposit}` : 'Deposit=10000',
    opts.currency ? `Currency=${opts.currency}` : 'Currency=USD',
    opts.report ? `Report=${opts.report}` : 'Report=mtcli_report',
    opts.visual ? 'Visual=1' : 'Visual=0',
    '',
    '[TesterInputs]',
    '',
  ].join('\n');
}

export function registerTerminalCommands(program: Command) {
  const term = program.command('terminal').description('Operações diretas no terminal (fora do listener)');

  term
    .command('start')
    .description('Inicia o terminal64.exe do projeto com chaves /config, /profile, /portable')
    .option('--config <ini>', 'Arquivo .ini customizado (/config)')
    .option('--profile <name>', 'Profile (/profile)')
    .option('--portable', 'Força modo portátil (/portable)', false)
    .option('--datapath <path>', 'Define /datapath:<path> se suportado')
    .option('--detach', 'Não esperar o processo (default: esperar)', false)
    .option('--project <id>', 'Projeto alvo')
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal) {
        throw new Error('terminal64.exe não configurado para o projeto.');
      }
      const args = buildArgs(opts);
      await runCommand(info.terminal, args, { stdio: opts.detach ? 'ignore' : 'inherit', detach: opts.detach });
      console.log(chalk.green(`[terminal] iniciado ${info.terminal} ${args.join(' ')}`));
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
    .option('--expert <name>')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--template <tpl>')
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
    .option('--model <n>', 'Modelagem (0=tick)', '0')
    .option('--spread <points>', 'Spread', '0')
    .option('--deposit <val>', 'Depósito', '10000')
    .option('--currency <ccy>', 'Moeda', 'USD')
    .option('--visual', 'Modo visual', false)
    .option('--report <name>', 'Nome base do report', 'mtcli_report')
    .action(async (opts) => {
      const content = renderTesterConfig(opts);
      const outPath = path.resolve(opts.out);
      await fs.writeFile(outPath, content, 'utf8');
      console.log(chalk.green(`[terminal] tester ini gerado em ${outPath}`));
    });

  term
    .command('launch')
    .description('Gera ini temporário (login/server/profile/expert) e abre terminal com /config')
    .requiredOption('--login <id>')
    .requiredOption('--password <pass>')
    .requiredOption('--server <srv>')
    .option('--profile <name>')
    .option('--expert <name>')
    .option('--symbol <symbol>')
    .option('--period <period>')
    .option('--template <tpl>')
    .option('--portable', 'Usar /portable', false)
    .option('--datapath <path>', 'Define /datapath:<path>')
    .option('--project <id>')
    .option('--keep-ini', 'Não apagar ini temporário', false)
    .action(async (opts) => {
      const info = await store.useOrThrow(opts.project);
      if (!info.terminal) throw new Error('terminal64.exe não configurado.');
      const tmpIni = path.join(process.cwd(), `mtcli_tmp_${Date.now()}.ini`);
      const content = renderConfigTemplate(opts);
      await fs.writeFile(tmpIni, content, 'utf8');
      const args = buildArgs({ ...opts, config: tmpIni });
      await runCommand(info.terminal, args, { stdio: 'inherit', detach: false });
      if (!opts.keepIni) await fs.remove(tmpIni).catch(() => {});
    });
}
