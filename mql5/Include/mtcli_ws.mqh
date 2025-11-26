//+------------------------------------------------------------------+
//| mtcli WebSocket helpers (cliente WS via Socket* nativo)          |
//| Inspirado no exemplo oficial do MQL5 (project_websocket_server)  |
//+------------------------------------------------------------------+
#property strict

// Pequena função para gerar chave; não validamos Accept no cliente.
string WS_BuildKey()
{
  // chave dummy base64 suficiente para servers permissivos
  return "dGhlIHNhbXBsZSBub25jZQ==";
}

bool WS_ClientHandshake(int sock, const string host, const string path="/")
{
  string key = WS_BuildKey();
  string req =
    "GET " + path + " HTTP/1.1\r\n"
    "Host: " + host + "\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Key: " + key + "\r\n\r\n";

  int sent = SocketSend(sock, req);
  if(sent != StringLen(req)) return false;

  uchar buf[2048];
  int r = SocketRead(sock, buf, 2047, 3000);
  if(r<=0) return false;
  buf[r]=0;
  string resp = CharArrayToString(buf,0,r);
  if(StringFind(resp, "101", 0) == -1) return false;
  if(StringFind(resp, "Sec-WebSocket-Accept", 0) == -1) return false;
  return true;
}

// Envia texto (opcode 0x1), cliente não mascara
bool WS_SendText(int sock, const string text)
{
  int len = StringLen(text);
  uchar frame[];
  if(len <= 125)
  {
    ArrayResize(frame, 2 + len);
    frame[0] = 0x81;
    frame[1] = (uchar)len;
    StringToCharArray(text, frame, 2, len);
  }
  else if(len <= 65535)
  {
    ArrayResize(frame, 4 + len);
    frame[0] = 0x81;
    frame[1] = 126;
    frame[2] = (uchar)((len >> 8) & 0xFF);
    frame[3] = (uchar)(len & 0xFF);
    StringToCharArray(text, frame, 4, len);
  }
  else
  {
    // não suportamos payloads enormes; aborta
    return false;
  }
  int sent = SocketSend(sock, frame);
  return sent == ArraySize(frame);
}

// Lê um frame texto (opcode 0x1) do servidor (sem máscara)
bool WS_ReadText(int sock, string &text, int timeoutMs=3000)
{
  uchar header[2];
  int h = SocketRead(sock, header, 2, timeoutMs);
  if(h != 2) return false;
  bool fin = (header[0] & 0x80) != 0;
  int opcode = header[0] & 0x0F;
  if(!fin || opcode != 1) return false;

  bool masked = (header[1] & 0x80) != 0;
  int payloadLen = header[1] & 0x7F;
  if(payloadLen == 126)
  {
    uchar ext[2];
    if(SocketRead(sock, ext, 2, timeoutMs) != 2) return false;
    payloadLen = (ext[0] << 8) | ext[1];
  }
  else if(payloadLen == 127)
  {
    // não suportamos >65535
    return false;
  }
  uchar mask[4];
  if(masked)
  {
    if(SocketRead(sock, mask, 4, timeoutMs) != 4) return false;
  }
  uchar payload[];
  ArrayResize(payload, payloadLen);
  int r = SocketRead(sock, payload, payloadLen, timeoutMs);
  if(r != payloadLen) return false;
  if(masked)
  {
    for(int i=0;i<payloadLen;i++) payload[i] = payload[i] ^ mask[i%4];
  }
  text = CharArrayToString(payload, 0, payloadLen);
  return true;
}
