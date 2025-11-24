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
}
