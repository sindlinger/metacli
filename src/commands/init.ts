import { Command } from 'commander';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectStore, repoRoot, ProjectDefaults } from '../config/projectStore.js';
import { deployFactoryConfig, deployFactoryTemplates, ensureCommandListenerStartup } from '../utils/factoryAssets.js';
import { ensureAccountInIni } from '../utils/factoryAssets.js';
import { normalizePath } from '../utils/paths.js';
import { restartListenerInstance } from './listener.js';
import { ensureCommandListenerCompiled, killTerminalIfRunning, killTerminalWindows, killTerminalByDatapath, startTerminalWindows, isTerminalRunning } from '../utils/commandListener.js';
import { provisionTerminalFromBase } from '../utils/terminalProvision.js';
import { sendListenerCommand } from '../utils/listenerProtocol.js';
import { collectAuthHints } from '../utils/logs.js';
import { softActivate } from './terminal.js';

const store = new ProjectStore();

const HARDCODED_DEFAULT_PROJECT = 'gpu72';

const DEFAULTS: ProjectDefaults = {
  symbol: 'EURUSD',
  period: 'M1',
  subwindow: 1,
  indicator: 'Examples\\ZigZag',
  expert: 'Examples\\MACD\\MACD Sample',
  portable: true,
  profile: null,
};

/**
 * init enxuto: registra um único terminal portátil e aplica assets de fábrica.
 * Sem parâmetros obrigatórios; usa caminhos conhecidos dentro do repositório.
 */
export function registerInitCommand(program: Command) {
  program
    .command('init')
    .description('Prepara projeto/terminal único (portátil) com arquivos de fábrica')
    .option('--project <id>', 'Nome do projeto (default gpu72)', HARDCODED_DEFAULT_PROJECT)
    .option('--data-dir <path>', 'Data_dir do terminal (default projects/terminals/<project>)')
    .action(async (opts) => {
      const projectId = opts.project || HARDCODED_DEFAULT_PROJECT;
      const defaultDir = path.join(repoRoot(), 'projects', 'terminals', projectId);
      const dataDir = normalizePath(opts.dataDir || defaultDir);
      const existing = (await store.load()).projects[projectId];
      if (existing) {
        console.log(
          chalk.yellow(
            `[init] Projeto "${projectId}" já existe. Use mtcli activate para reconfigurar e relançar.`
          )
        );
        return;
      }
      await fs.ensureDir(dataDir);

      // provisiona terminal se não existir (copia de base conhecida)
      await provisionTerminalFromBase(dataDir);
      const terminal = path.join(dataDir, 'terminal64.exe');
      const metaeditor = path.join(dataDir, 'MetaEditor64.exe');
      const libs = path.join(dataDir, 'MQL5', 'Libraries');

      // Cria estrutura mínima (factory) se faltando
      await fs.ensureDir(libs).catch(() => {});
      await deployFactoryConfig(dataDir);
      await deployFactoryTemplates(dataDir);
      await ensureCommandListenerStartup(dataDir, DEFAULTS);
      await ensureAccountInIni(dataDir).catch(() => {}); // opcional (ducascopy creds se existirem)
      await ensureCommandListenerCompiled(dataDir).catch((err) => {
        console.log(chalk.yellow(`[init] Não consegui compilar CommandListenerEA: ${err.message}`));
      });

      // origin.txt
      const originFile = path.join(dataDir, 'origin.txt');
      if (!(await fs.pathExists(originFile))) {
        await fs.writeFile(
          originFile,
          `project=${projectId}\ncreated=${new Date().toISOString()}\npath=${dataDir}\n`,
          'utf8'
        );
      }

      // Salva no mtcli_projects.json
      await store.setProject(
        projectId,
        {
          project: projectId,
          data_dir: dataDir,
          terminal,
          metaeditor,
          libs,
          defaults: { ...DEFAULTS },
        },
        true
      );

      console.log(chalk.green(`[init] Projeto ${projectId} registrado com data_dir=${dataDir}`));
      console.log(chalk.gray('[init] Aplicando verify --fix para garantir estrutura...'));

      // chamar verify --fix implicitamente? preferir garantir aqui:
      // (já garantimos config/templates; apenas cria pastas se faltarem)
      const folders = [
        'Bases',
        'Config',
        'Logs',
        'MQL5',
        'Profiles',
        'Templates',
        'Tester',
        path.join('MQL5', 'Experts'),
        path.join('MQL5', 'Indicators'),
        path.join('MQL5', 'Scripts'),
        path.join('MQL5', 'Include'),
        path.join('MQL5', 'Files'),
        path.join('MQL5', 'Images'),
        path.join('MQL5', 'Libraries'),
        path.join('MQL5', 'Logs'),
        path.join('MQL5', 'Presets'),
        path.join('MQL5', 'Profiles', 'Charts'),
        path.join('MQL5', 'Profiles', 'Templates'),
        path.join('MQL5', 'Profiles', 'SymbolSets'),
        path.join('MQL5', 'Profiles', 'Tester'),
        path.join('Tester', 'logs'),
        path.join('Tester', 'Cache'),
      ];
      for (const f of folders) {
        await fs.ensureDir(path.join(dataDir, f));
      }

      console.log(chalk.green('[init] Estrutura mínima garantida.'));
      console.log(chalk.gray('[init] Reiniciando terminal em background para carregar CommandListener...'));
      await killTerminalIfRunning(terminal).catch(() => {});
      await killTerminalWindows(terminal).catch(() => {});
      await killTerminalByDatapath(dataDir).catch(() => {});
      await startTerminalWindows(terminal, dataDir).catch(async () => {
        await restartListenerInstance({ project: projectId }).catch(() => {});
      });
      await new Promise((r) => setTimeout(r, 6000));
      const running = await isTerminalRunning(terminal);
      if (!running) {
        console.log(
          chalk.yellow(
            '[init] Processo não confirmado após o start. Se a janela não estiver aberta, execute o run-terminal.ps1 no data_dir; depois rode mtcli events ping.'
          )
        );
      }
      const authHints = collectAuthHints(dataDir, 40);
      if (authHints.length) {
        console.log(chalk.gray('[init] Pistas de autenticação nos logs:'));
        console.log(authHints.join('\n'));
      }
      // tenta PING, mas sem alegar sucesso se offline
      const tryPing = async () => {
        await sendListenerCommand(
          {
            project: projectId,
            terminal,
            data_dir: dataDir,
            defaults: DEFAULTS,
          } as any,
          'PING',
          [],
          { timeoutMs: 3000, ensureRunning: false, allowRestart: false }
        );
      };
      try {
        await tryPing();
        console.log(chalk.green('[init] CommandListener online (PING ok).'));
      } catch {
        console.log(
          chalk.yellow(
            '[init] CommandListener não respondeu. Após abrir o terminal, rode mtcli open-terminal ou mtcli events ping para reanexar/checar.'
          )
        );
      }
      console.log(chalk.green('[init] Finalizado.'));
    });
}
