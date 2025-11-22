# AGENTS.md — mtcli-integrado + Reversal Wave GPU

Este repositório define um ambiente de trabalho para o Codex resolver problemas
de MT5/MQL5 usando o CLI `mtcli` e, especificamente, implementar e evoluir
um pipeline de **reversal wave** baseado em FFT/conv/corr em **C++/GPU**, exposto
para MQL5 apenas via DLL.

Este arquivo é um **documento de contrato para agentes**. Ele NÃO é código
executável, mas descreve:

- quais agentes existem;
- qual é o escopo de cada agente;
- quais pastas/arquivos fazem parte de cada fluxo;
- o que PODE e o que NÃO PODE ser feito em MQL5;
- o que TEM que ser feito em C++ (DLL, GPU).

O Codex (ou outro orquestrador) pode simplesmente carregar este arquivo como
contexto inicial quando estiver trabalhando neste repositório.

---

## 1. Visão geral do repositório

Layout relevante (simplificado):

- `src/`  
  Código TypeScript/Node do CLI `mtcli`.

- `dist/`  
  Saída compilada (JS) do CLI.

- `bin/mtcli` / `bin/mtcli.cmd`  
  Entradas principais do CLI para Unix/Windows.

- `agent/`  
  Arquivos de configuração de agente (`agent_config.yaml`, `tools_index.yaml`,
  `knowledge_index.yaml`) que apontam para:
  - prompts em `prompts/`,
  - docs em `docs/`,
  - e o próprio `mtcli`.

- `prompts/`  
  Arquivos YAML de comportamento, ferramentas e fluxos de trabalho.

- `docs/`  
  Documentação local (MQL5, MT5, etc).

- `native-fft/`  
  Camada nativa/C++ onde vive o pipeline de FFT/conv/corr.  
  Aqui é onde o **reversal wave GPU/CPU pipeline** deve ser implementado:
  - `native-fft/include/fasttransforms.h`
  - `native-fft/src/fasttransforms.cpp` :contentReference[oaicite:0]{index=0}

- `mql5/` (planejado/organizado por este agente)  
  Shells MQL5 que apenas:
  - coletam dados (Close, Volume, pivôs, etc.),
  - chamam a DLL,
  - desenham no gráfico.

---

## 2. Agente “mtcli-integrated” (geral do repositório)

### Nome lógico

`mtcli-integrated`

### Objetivo

Agir como **agente principal** do repositório, capaz de:

- entender a estrutura do `mtcli` (src, dist, bin, tools, scripts);
- usar o `mtcli` para:
  - compilar EAs/indicadores;
  - rodar testers;
  - anexar/desanexar indicadores/EAs em gráficos;
- consultar documentação local de MQL5/MT5 e docs do próprio CLI;
- manter e melhorar o `mtcli` (código e docs), sempre de forma compatível com
  o estilo existente.

### Regras principais

1. **mtcli é a autoridade** para interação com o MT5.
   - Nunca reinventar scripts externos redundantes se o `mtcli` já cobre o caso.
   - Descobrir sintaxe via `mtcli --help` e `mtcli <comando> --help`.

2. **Documentação local primeiro**:
   - usar `docs/`, `legacy_docs/`, `mt_agent_pack/`, `mt_agent_recipes/`,
     `AGENTS.md` e `README.md` antes de sair para web search.

3. **MQL5**:
   - respeitar as regras de arrays em série temporal (index 0 = barra atual),
     `rates_total`, `prev_calculated` e buffers de indicador;
   - NUNCA misturar lógica pesada de cálculo FFT/conv/corr em MQL5 quando existir
     uma DLL nativa prevista.

4. **CLI simples para o Codex**:
   - idealmente, o Codex deve conseguir operar tudo com comandos óbvios,
     como `mtcli init`, `mtcli project ...`, `mtcli tester ...`;
   - evitar superfícies complexas que requeiram conhecimento “oculto” do CLI.

---

## 3. Agente “reversal-wave-gpu” (pipeline FFT/conv em DLL)

### Nome lógico

`reversal-wave-gpu`

### Objetivo

Construir e evoluir um pipeline numérico robusto (FFT/conv/corr) em C++/GPU
dentro da DLL nativa, de forma que:

- **toda a matemática pesada** (FFT, filtros, convoluções, correlações, windowing,
  detrend, normalização, métricas de confiança) fique na DLL;
- o código MQL5 fique **ultraleve**, servindo apenas de “cola”:
  - coleta de dados,
  - chamada à DLL,
  - desenho no gráfico;
- o resultado seja uma ou mais “waves” que representem potenciais pontos de
  reversão de preço (bull/bear), com uma noção de confiança.

### Escopo de arquivos

O agente deve trabalhar principalmente em:

- **DLL / C++ da pipeline**  
  - `native-fft/include/fasttransforms.h`
  - `native-fft/src/fasttransforms.cpp`

- **Bridge MQL5 (cola)**  
  - `mql5/Include/ReversalWaveBridge.mqh` (a criar/ajustar)
  - `mql5/Indicators/ReversalWave.mq5` (a criar/ajustar)
  - `mql5/Scripts/ReversalWaveSelfTest.mq5` (opcional mas recomendado)

- **Documentação do pipeline**  
  - `docs/reversal_wave_pipeline_overview.md` (a criar/ajustar)

### Contrato da API da DLL

#### Função principal: `gpu_reversal_wave_process`

Entradas:

- `price[i]` — série de preço (ex.: `Close`), em **ordem cronológica**  
  (do mais antigo para o mais recente).

- `volume[i]` — série de volume, mesma ordem.

- `pivots[i]` — codificação discreta dos pivôs:
  - `+1` para topo,
  - `-1` para fundo,
  - `0` para barra “normal”.

- `length` — tamanho das séries (`price/volume/pivots`).

- `window` — tamanho da janela de processamento (em barras), ex.: 128, 192, 256.

- `modeFlags` — flags combinadas (OR bit a bit) para modos da pipeline, por ex.:
  - `kModeHighPass`
  - `kModeEmphasizePivot`
  - `kModeUseHannWindow`
  - `kModeBandPassFibo` (se futuramente existir um modo de banda baseado em
    números de Fibonacci)
  - etc.

- `priceWeight`, `volumeWeight`, `pivotWeight` — pesos relativos de cada canal,
  ex.: `0.7`, `0.2`, `0.1`.

Saídas:

- `outWave[i]` — valor contínuo da wave (normalmente centrada em 0) para cada
  barra ou, pelo menos, para as barras mais recentes de interesse.

- `outConfidence[i]` — confiança (0.0–1.0) do sinal naquela barra, levando em
  conta coerência de fase, energia de banda, qualidade do detrend, etc.

- `outFlags[i]` — flags discretas (bitmask) indicando:
  - possível reversão de alta (bull),
  - possível reversão de baixa (bear),
  - zona de saturação / ruído,
  - qualquer outro status relevante.

Retorno (`int`):

- `0` → sucesso;
- valores negativos → erro:
  - argumento inválido,
  - tamanho insuficiente,
  - falha interna, etc.

#### Função auxiliar: `gpu_reversal_wave_synthetic_test`

Objetivo:

- gerar internamente uma série sintética simples (ex.: senoide + ruído),
- passar pela **mesma** pipeline de FFT/conv/detrend/filtros,
- escrever `outWave`, `outConfidence`, `outFlags` para validação rápida.

Uso típico:

- chamada via script MQL5 `ReversalWaveSelfTest.mq5`;
- serve para testar a DLL sem depender de dados de mercado reais;
- bom para sanity checks e testes de regressão.

---

## 4. Regras rígidas para este agente

### 4.1. O que DEVE ficar em C++/DLL (GPU/CPU nativa)

Pode / deve ser implementado (ou mantido) em C++ nativo, com foco em GPU:

- detrend (remover componente de tendência / nível DC / drift);
- normalização (escala de amplitude razoável, evitar overflow/underflow);
- construção de sinal composto a partir de:
  - preço (ex.: série detrended do Close),
  - volume (ex.: volume normalizado, ou algum realce de anomalias),
  - pivôs (ZigZag, padrões 1-2-3, etc., codificados em +1/-1/0);

- FFT / IFFT (real/complex) e variações (FHT se fizer sentido);
- filtros de banda (high-pass, band-pass, notch, etc.);
- convoluções e correlações (incluindo as variantes circulares e não circulares);
- janelas (Hann, Hamming, Blackman, etc.);
- cálculo de métricas de confiança do sinal;
- mapeamento para um sinal discreto de reversão:
  - bull/bear,
  - força do sinal (fraca/forte),
  - flags adicionais.

**Regra de ouro:**  
Se o algoritmo é naturalmente O(N log N), O(N²) ou demanda FFT/conv/corr, ele
pertence à DLL, não ao MQL5.

### 4.2. O que PODE ficar em MQL5 (mas só cola)

Em MQL5, o agente pode (e deve) fazer apenas:

- coletar arrays:
  - `Close[]`, `Open[]`, `High[]`, `Low[]`, `Volume[]`, etc.;
  - buffers de ZigZag, pivôs, padrões 1-2-3 (se existirem como indicadores
    auxiliares);
- montar arrays usados pela DLL (já respeitando ordem cronológica, série temporal,
  etc.);
- chamar `gpu_reversal_wave_process` e `gpu_reversal_wave_synthetic_test`
  via `#import` em `ReversalWaveBridge.mqh`;
- copiar resultados (`outWave`, `outConfidence`, `outFlags`) para buffers
  de indicador;
- desenhar:
  - wave como linha/curva,
  - confiança como histograma ou linha auxiliar,
  - setas/labels em potenciais reversões.

Não pode em MQL5:

- reimplementar FFT/conv/corr;
- fazer pipelines numéricas grandes tipo “mini-DSP”;
- duplicar lógica que já está ou deveria estar na DLL.

### 4.3. Estilo de desenvolvimento

No C++ / DLL:

- checar todos os argumentos de entrada:
  - ponteiros nulos,
  - tamanhos inconsistentes,
  - pesos com soma bizarra, etc.;
- proteger contra `NaN` / `INF` – se encontrar, retornar erro claro;
- manter código modular:
  - funções internas bem nomeadas, por exemplo:
    - `build_composite_signal(...)`,
    - `apply_detrend_and_window(...)`,
    - `apply_fft_bandpass(...)`,
    - `compute_confidence_and_flags(...)`,
    - etc.;
- projetar o código pronto para GPU (CUDA/OpenCL) mesmo que no começo rode
  em CPU (organizar separando “pipeline lógico” da “backend de execução”).

No MQL5:

- seguir o contrato de `OnCalculate` (`rates_total`, `prev_calculated`);
- tratar corretamente arrays em série temporal (`ArraySetAsSeries`);
- evitar repintar sinais passados (exceto onde isso faz sentido e está
  explicitamente documentado);
- manter indicadores focados: um indicador para a wave e suas confidências,
  sem misturar dezenas de responsabilidades.

---

## 5. Workflow recomendado para o Codex

### 5.1. Edição e testes da DLL

1. Ler `docs/reversal_wave_pipeline_overview.md` para entender o desenho da
   pipeline (se ainda não existir, o agente deve criá-lo).

2. Trabalhar em:

   - `native-fft/include/fasttransforms.h`
   - `native-fft/src/fasttransforms.cpp`

   Aproveitando as rotinas já existentes de FFT/conv/corr em ALGLIB, em vez
   de reinventar a roda. :contentReference[oaicite:1]{index=1}

3. Compilar a DLL (via CMake/projeto de IDE/script já existente no repo).

4. Copiar a DLL resultante para `MQL5/Libraries` na instalação alvo do MT5
   dedicada ao agente.

5. Usar o script `ReversalWaveSelfTest.mq5` para um sanity check rápido.

6. Depois, anexar o indicador `ReversalWave.mq5` a um gráfico real e
   inspecionar:

   - forma da wave,
   - coerência com topos/fundos e pivôs,
   - comportamento da confiança.

### 5.2. Integração MQL5 detalhada

1. **Bridge** — `mql5/Include/ReversalWaveBridge.mqh`:

   - declarar o `#import` da DLL (nome a ser definido, ex.: `"ReversalWave.dll"`),
   - mapear:
     - `gpu_reversal_wave_process(...)`,
     - `gpu_reversal_wave_synthetic_test(...)`;
   - expor uma função helper:
     - `bool RWComputeWave(...)` que:
       - monta as chamadas,
       - trata códigos de erro,
       - escreve logs de erro mínimos.

2. **Indicador** — `mql5/Indicators/ReversalWave.mq5`:

   - parâmetros de entrada:
     - pesos (`priceWeight`, `volumeWeight`, `pivotWeight`),
     - `window`,
     - flags de modo (checkbox/enum);
   - coletar dados de preço/volume/pivôs;
   - chamar `RWComputeWave` em `OnCalculate`;
   - preencher buffers:
     - wave,
     - confiança,
     - flags discretas (ex.: histograma bull/bear);
   - desenhar wave e sinais de reversão.

3. **Script de teste** — `mql5/Scripts/ReversalWaveSelfTest.mq5`:

   - chamar `gpu_reversal_wave_synthetic_test(...)`
     com parâmetros tipo:
     - `length`,
     - `oscillation`,
     - `noiseLevel`;
   - logar no Journal/Experts:
     - alguns valores da wave,
     - estatísticas simples,
     - se houver flags de reversão.

---

## 6. Como este AGENTS.md será usado

- Ao trabalhar no repositório `mtcli-integrado`, o orquestrador (Codex) deve
  sempre carregar este `AGENTS.md` como parte do contexto inicial.
- O agente “geral” (`mtcli-integrated`) cuida de tudo que é CLI/MT5/MQL5.
- O agente “especialista” (`reversal-wave-gpu`) foca **exclusivamente** no
  pipeline FFT/conv/detrend em DLL e na cola MQL5 mínima necessária.

Ambos devem respeitar as regras:

- nada de duplicar lógica pesada em MQL5;
- nada de APIs ocultas ou superfícies confusas no `mtcli`;
- tudo que for importante para o futuro deve ser documentado (idealmente em:
  - `AGENTS.md`,
  - `docs/reversal_wave_pipeline_overview.md`,
  - comentários claros no C++ e no MQL5 de bridge/indicador/script).
