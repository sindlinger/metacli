import { Command } from 'commander';
import fs from 'fs';
import chalk from 'chalk';
import path from 'path';
import { execa } from 'execa';
import os from 'os';
import { ProjectStore, requireField } from '../config/projectStore.js';
import { normalizePath, toWslPath } from '../utils/paths.js';
import { resolveMetaeditorArgs } from '../utils/metaeditor.js';
import { resolveTarget } from '../utils/target.js';

/**
 * Compila um arquivo .mq5/.mqh usando o MetaEditor configurado no projeto ativo.
 * Foi adicionado para substituir o "atalho" que aparecia no help mas não existia.
 */
export function registerCompileCommands(program: Command) {
  program
    .command('compile')
    .description('Compila um .mq5/.mqh com o MetaEditor do projeto ativo (sem parâmetros obrigatórios)')
    .option('--file <path>', 'Caminho do arquivo .mq5/.mqh (absoluto ou relativo)')
    .option('-i, --indicator <name>', 'Nome/atalho do indicador (pode ser I:ZigZag, ZigZag, subpasta)')
    .option('-e, --expert <name>', 'Nome/atalho do expert (pode ser E:MyEA, MyEA, subpasta)')
    .option('--log <path>', 'Salvar log do MetaEditor neste arquivo')
    .option('--project <id>', 'Projeto alvo (padrão: last_project)')
    .action(async (opts) => {
      const store = new ProjectStore();
      const info = await store.useOrThrow(opts.project);
      const meta = requireField(info.metaeditor, 'MetaEditor não configurado no projeto.');
      const target = resolveTarget(info, { file: opts.file, indicator: opts.indicator, expert: opts.expert });
      const fileAbs = target.file;
      if (!fs.existsSync(fileAbs)) {
        throw new Error(`Arquivo não encontrado: ${fileAbs}`);
      }
      const logPath = opts.log ? normalizePath(opts.log) : undefined;
      if (logPath) {
        const logDir = path.dirname(logPath);
        if (!fs.existsSync(logDir)) {
          fs.mkdirSync(logDir, { recursive: true });
        }
      }
      const args = resolveMetaeditorArgs(info, fileAbs, logPath);
      const metaExec = os.platform() === 'linux' ? toWslPath(meta) : meta;
      console.log(chalk.gray(`[compile] ${metaExec} ${args.join(' ')}`));
      const res = await execa(metaExec, args, { stdio: 'inherit', windowsHide: false, reject: false });
      if (res.exitCode !== 0) {
        console.log(chalk.yellow(`[compile] MetaEditor retornou exitCode=${res.exitCode} (pode ser normal no WSL).`));
      }
      console.log(chalk.green('[compile] sucesso'));
    });
}
