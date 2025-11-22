==================== AGENTS.md ====================

# Agents registry (mtcli-integrado)

Este arquivo descreve os agentes que podem atuar neste repositório.
O que importa aqui é orientar o modelo sobre:

- **onde** mexer (paths),
- **o que** pode / não pode fazer em cada camada,
- **qual task file** usar como prompt base.

---

## Agent: reversal_wave_gpu_pipeline

### Nome interno

`reversal_wave_gpu_pipeline`

### Objetivo

Construir e manter uma pipeline de **Reversal Wave** orientada a pivôs (ZigZag + padrão 1‑2‑3), rodando em **DLL nativa com caminho para GPU**, com o MT5/MQL5 servindo apenas como camada de integração (coleta de dados e desenho no gráfico).

A wave deve:

- ser centrada em torno de zero (detrend/normalizada),
- destacar **pontos de reversão** (topos/fundos) em múltiplas escalas,
- combinar **preço, volume e pivôs** (ZigZag + candles 1‑2‑3),
- expor um **sinal contínuo** (outWave),
- uma **confiança [0..1]** por barra (outConfidence),
- e um **bitmask discreto de flags** (outFlags) com:
  - `BULLISH` (provável fundo / reversão altista),
  - `BEARISH` (provável topo / reversão baixista),
  - `LOW_CONFIDENCE` (sinal fraco / ruidoso),
  - `WARMUP` (barras iniciais sem confiabilidade).

### Task file principal

- `TASK_REVERSAL_WAVE_GPU_PIPELINE.txt`

Este task file é o contrato operacional desse agente: descreve o API da DLL, as regras rígidas de GPU vs MQL5, o pipeline numérico de detrenching/FFT, e os critérios de “pronto”.

### Layout do repositório (relevante para este agente)

- **Código nativo (DLL / GPU-friendly)**
  - `native-fft/include/fasttransforms.h`
  - `native-fft/src/fasttransforms.cpp`
  - (outros arquivos `.cpp/.h` relacionados à DLL podem ser usados se necessário,
    mas **não criar** APIs paralelas para a mesma wave; centralizar no contrato único)

- **Integração com MT5 (cola MQL5)**
  - `mql5/Include/ReversalWaveBridge.mqh`
    - `#import` da DLL
    - helpers como `RWComputeWave(...)`, `RWIsBullish(...)`, etc.
  - `mql5/Indicators/ReversalWave.mq5`
    - indicador que:
      - coleta Close[], Volume[] e pivôs (ZigZag + 1‑2‑3),
      - chama o bridge da DLL,
      - recebe outWave/outConfidence/outFlags,
      - desenha tudo no gráfico (linha contínua, histograma de confiança, setas bull/bear).
  - `mql5/Scripts/ReversalWaveSelfTest.mq5`
    - script de autoteste:
      - chama `gpu_reversal_wave_synthetic_test(...)`,
      - grava no Journal contagens de bull/bear/low_confidence/warmup.

- **Documentação**
  - `docs/reversal_wave_pipeline_overview.md`
    - descrição da pipeline: detrend, FFT, filtros, montagem de sinal, flags, etc.

### Contrato da API da DLL (resumo)

Função principal:

```c
int gpu_reversal_wave_process(
    const double* price,
    const double* volume,
    const double* pivots,
    int length,
    int window,
    int modeFlags,
    double priceWeight,
    double volumeWeight,
    double pivotWeight,
    double* outWave,
    double* outConfidence,
    int* outFlags);

Entradas

price[i] – série de preço (p. ex. Close), em ordem cronológica (mais antigo → mais recente).

volume[i] – série de volume, mesma ordem.

pivots[i] – codificação dos pivôs:

+1 topo, -1 fundo, 0 barra “normal” (ZigZag + padrão 1‑2‑3 se desejado).

length – tamanho das séries.

window – tamanho da janela de processamento (em barras).

modeFlags – combinação (bitwise OR) de flags de modo da pipeline:

ex.: kModeHighPass, kModeEmphasizePivot, kModeUseHannWindow, etc.

priceWeight, volumeWeight, pivotWeight – pesos relativos de cada canal.

Saídas

outWave[i] – valor contínuo da wave (centrado em 0, detrend/normalizado).

outConfidence[i] – confiança (0.0–1.0) do sinal naquela barra.

outFlags[i] – bitmask discreto:

kFlagBullish – provável fundo / reversão de compra.

kFlagBearish – provável topo / reversão de venda.

kFlagLowConfidence – sinal fraco / ruidoso.

kFlagWarmup – barras em aquecimento (janela inicial).

Retorno (int)

0 – sucesso.

< 0 – erro:

invalid argument, tamanho insuficiente, falha interna, etc.

Função auxiliar de teste sintético:

int gpu_reversal_wave_synthetic_test(
    int length,
    double oscillation,
    double noiseLevel,
    double* outWave,
    double* outConfidence,
    int* outFlags);


Gera internamente uma série sintética (senoidal + ruído) e roda a mesma pipeline.

Usada pelo script ReversalWaveSelfTest.mq5 para validar a DLL sem precisar de dados do mercado.

Regras rígidas para este agente
O que pode / deve ir para C++ (DLL / GPU)

Detrend e normalização (subtração de tendência, remoção de média linear, scaling).

Construção de sinal composto:

combinação de preço, volume, pivôs, padrões 1‑2‑3 de candle.

FFT/IFFT, Fast Hartley Transform, filtros de banda, janelas (Hann, Hamming etc.).

Convoluções e correlações (inclusive circulares) para “esculpir” a wave.

Cálculo de outConfidence.

Lógica de detecção de reversão e geração de bitmask outFlags:

decidir quais barras são bull/bear,

marcar warmup,

marcar low confidence.

Sanitização de séries de entrada (NaN/Inf → 0 ou comportamento seguro).

Qualquer outro processamento O(N log N) ou mais pesado.

O que PODE ir para MQL5 (apenas cola)

Coletar dados do MT5: Close[], Volume[], buffers do ZigZag, padrões 1‑2‑3.

Converter esses dados nos arrays price, volume, pivots esperados pela DLL.

Chamar a DLL via #import (bridge em ReversalWaveBridge.mqh).

Copiar outWave, outConfidence, outFlags para buffers de indicador.

Desenhar no gráfico:

linha da wave,

histograma de confiança,

setas ou ícones de bull/bear,

labels de debug opcional.

O que NÃO pode ir para MQL5

FFT, FHT, convolução, correlação de qualquer tipo.

Detrend “inteligente” (regressão, band-pass, etc.).

Pipeline de cálculo pesada (loops O(N log N), O(N²), etc.), salvo coisinhas triviais.

Tentativas de “reimplementar” a wave em puro MQL.
👉 Toda inteligência numérica pesada é na DLL.

Estilo de desenvolvimento esperado

Código C++:

limpo, modular, com funções internas bem nomeadas,

checagem de argumentos de entrada (null, tamanhos, ranges),

tolerante a dados ruins (NaN/Inf, buracos, spikes),

pronto para ser portado para GPU (CUDA/OpenCL) no futuro:

data layout contíguo,

minimizar alocação por chamada,

evitar dependências não-portáveis.

Código MQL5:

indicadores focados em uma coisa só (esta wave),

buffers e índices consistentes com rates_total / prev_calculated,

sem lógica duplicada do lado MQL (usar sempre a DLL).

“Done when” – o que significa “wave pronta exibindo e afinada”

O agente só deve considerar a task concluída quando:

A DLL compila sem erros e exporta:

gpu_reversal_wave_process,

gpu_reversal_wave_synthetic_test.

O script ReversalWaveSelfTest.mq5:

executa sem erro,

loga contagens coerentes de bullish, bearish, low_confidence, warmup,

não gera explosões de NaN/Inf (séries sanitizadas).

O indicador ReversalWave.mq5:

pode ser anexado a um gráfico real,

desenha a wave centrada em 0,

desenha a confiança,

desenha pontos de bull/bear usando outFlags da DLL (não heurísticas locais),

responde de forma estável a diferentes símbolos/tempos (não “explode” ao trocar timeframe).

O pipeline está “afinável”:

existem inputs (no indicador ou em bridge) para ajustar:

window,

modeFlags,

priceWeight, volumeWeight, pivotWeight,

esses ajustes de fato modificam o comportamento da wave de forma suave,

documentação mínima em docs/reversal_wave_pipeline_overview.md explica:

significado de cada parâmetro,

interpretação de outWave/outConfidence/outFlags.
