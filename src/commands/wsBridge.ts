import { Command } from 'commander';
import chalk from 'chalk';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';

interface Client {
  ws: WebSocket;
  id: string;
}

export function registerWsBridgeCommand(program: Command) {
  program
    .command('ws-bridge')
    .description('Inicia um bridge WebSocket entre o mtcli e o CommandListener (canal rápido, sem Files)')
    .option('-p, --port <n>', 'Porta para escutar', '8787')
    .action(async (opts) => {
      const port = Number(opts.port) || 8787;
      const server = createServer();
      const wss = new WebSocketServer({ server });
      let listenerClient: Client | null = null;

      wss.on('connection', (ws: WebSocket, req) => {
        const clientId = `${req.socket.remoteAddress || 'unknown'}:${req.socket.remotePort || ''}`;
        console.log(chalk.gray(`[ws] conexão de ${clientId}`));

        ws.on('message', (data) => {
          try {
            const msg = JSON.parse(data.toString());
            if (msg?.role === 'listener') {
              listenerClient = { ws, id: clientId };
              ws.send(JSON.stringify({ ok: true, msg: 'listener-registered' }));
              console.log(chalk.green('[ws] listener registrado'));
              return;
            }
            if (msg?.type && listenerClient && msg?.role === 'user') {
              listenerClient.ws.send(JSON.stringify(msg));
              return;
            }
          } catch (err) {
            console.log(chalk.red(`[ws] erro parse ${err}`));
          }
        });

        ws.on('close', () => {
          if (listenerClient && listenerClient.ws === ws) {
            console.log(chalk.yellow('[ws] listener desconectado'));
            listenerClient = null;
          } else {
            console.log(chalk.gray(`[ws] cliente ${clientId} desconectado`));
          }
        });
      });

      server.listen(port, () => {
        console.log(chalk.green(`[ws] bridge escutando em ws://0.0.0.0:${port}`));
        console.log(chalk.gray('Adicione este host/porta na lista de WebRequest do MT5.'));
      });
    });
}
