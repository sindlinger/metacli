# AGENTS.md — mtcli-integrado

Este arquivo descreve **como o Codex deve atuar dentro deste repositório**  
(`/mnt/e/dev/mtcli/mtcli-integrado`) e, em particular, define o agente
**REVERSAL_WAVE_GPU_PIPELINE**, responsável por projetar e manter o pipeline
de GPU para detecção de reversões.

---

## 1. Arquitetura geral deste projeto

Visão de alto nível:

- **Codex (CLI `codex`)**
  - Orquestrador inteligente.
  - Lê tarefas (`TASK_*.txt`), lê este `AGENTS.md`, navega pelo repo,
    edita código, roda testes via `mtcli` quando necessário.

- **mtcli (CLI Node/TypeScript deste repositório)**
  - Interface principal com o MetaTrader 5:
    - compilar indicadores/EAs,
    - instalar na pasta de usuário,
    - acionar Strategy Tester,
    - anexar/destacar indicadores/EAs em gráficos, etc.
  - Codex deve tratar o `mtcli` como ferramenta de automação, **não**
    como lugar para enfiar lógica numérica pesada.

- **native-fft (DLL nativa + GPU)**
  - Diretório para **código C++/GPU**:
    - FFT (via ALGLIB / `fasttransforms`),
    - convoluções/correlações,
    - filtros de frequência,
    - toda a lógica de **wave de reversão**.
  - O alvo é rodar tudo que for pesado na GPU (ex.: CUDA, OpenCL, etc.),
    mas a API deve continuar exposta como DLL padrão para o MT5.

- **Indicadores MQL5 (.mq5 / .mqh)**
  - Apenas:
    - capturam dados do gráfico (preço, volume, zigzag, padrões 1‑2‑3, etc.),
    - empacotam arrays,
    - fazem **chamadas para a DLL** (GPU),
    - desenham o resultado no gráfico.
  - **Sem lógica numérica séria aqui.** Nenhum processamento de FFT,
    convoluções, agregações de cluster, etc. Isso pertence à DLL.

---

## 2. Regras de projeto (hard constraints)

Quando o Codex estiver atuando neste repositório, ele deve seguir **sempre**:

1. **Toda computação pesada na DLL (GPU)**  
   - FFT, convoluções, correlações, filtros, síntese de waves, etc.
   - Devem morar em `native-fft/` (C++/GPU), usando as rotinas numéricas
     disponíveis (`fftr1d`, `fftr1dinv`, `convr1d`, `corrr1d`, etc.).:contentReference[oaicite:1]{index=1}  

2. **MQL5 (.mq5/.mqh) sem implementação de processo**
   - Permitido:
     - declaração de `import` da DLL,
     - structs simples para transportar dados,
     - cópia de dados das séries (Open, High, Low, Close, Volume, Time),
     - desenho no gráfico (objetos, buffers de indicador).
   - **Não permitido**:
     - laços intensivos sobre barras implementando algoritmos de detecção,
     - FFT, convolução, correlação, clustering, estatística pesada,
     - lógica de decisão complexa (isso é responsabilidade da DLL).

3. **O indicador é apenas “casca”**
   - OnInit:
     - inicializa buffers,
     - configura estilos de plot,
     - inicializa ponte com a DLL (se necessário).
   - OnCalculate:
     - prepara arrays de entrada,
     - chama a função da DLL,
     - recebe arrays de saída já processados (waves, scores, flags),
     - escreve nos buffers para plot.

4. **GPU em primeiro lugar**
   - O Codex deve:
     - estruturar o código C++ de forma **amigável à GPU** (kernel functions,
       memória contígua, bateladas, etc.),
     - evitar cópias desnecessárias host↔device,
     - concentrar a latência numa chamada relativamente grossa
       (por ex. “processa N barras de uma vez”).

5. **Nenhuma duplicação de lógica entre MQL5 e C++**
   - A regra é: **se uma fórmula existe na DLL, o MQL5 não recalcula.**
   - Se for preciso adaptar algo, adapta‐se a DLL.

---

## 3. Agente: REVERSAL_WAVE_GPU_PIPELINE

### 3.1 Nome e papel

**Nome interno do agente:** `reversal_wave_gpu_pipeline`  

**Missão:**  
Projetar, implementar e manter um pipeline de GPU que:

- recebe múltiplos sinais (preço, volume, pivôs, padrões de candle, etc.),
- transforma isso em uma ou mais “waves” que sintetizam zonas prováveis de
  reversão (topos/fundos),
- expõe uma API de DLL simples para MQL5,
- permite que o indicador **só desenhe** e gerencie entrada/saída.

### 3.2 Sinais de entrada esperados

O agente deve assumir (e, se necessário, padronizar) algo nessa linha:

- **Preço**
  - Série Close (ou média típica, ou mediana) de N barras.
  - Pré-processamento:
    - opção de **detrend** (subtrair reta de regressão ou média suave),
    - normalização (ex.: dividir pelo desvio padrão local).

- **Volume**
  - Série ligada às mesmas N barras.
  - Normalização típica:
    - volume relativo ao volume médio da janela,
    - ou log(volume / média).

- **Estrutura de pivôs**
  - Pivôs vindos de um **ZigZag** e/ou de padrão **1‑2‑3**:
    - um vetor indicando, por barra, se é topo, fundo ou nada,
    - amplitude dos swings, se disponível.

- **Outras features opcionais**
  - volatilidade local (ATR ou range),
  - cluster PRZ já calculado (se você reaproveitar algo do indicador antigo),
  - flags de “pivô confirmado” (por ex. 2 barras depois).

### 3.3 Pipeline conceitual (no C++/GPU)

O agente deve buscar uma estrutura de pipeline deste tipo, **inteira na DLL**:

1. **Empacotamento de sinais**
   - Constrói um vetor/espaço multi‑canal:
     - canal 0: preço detrended,
     - canal 1: volume normalizado,
     - canal 2: intensidade de pivô/topo/fundo,
     - (opcionais: canais extra de volatilidade, PRZ, etc.).

2. **Janela / windowing**
   - Aplicar janela (Hann/Hamming ou similar) por bloco de N barras
     antes da FFT, se necessário, para evitar artefatos.

3. **Transformada (FFT)**
   - Usar FFT real ou complexa (`fftr1d`, `fftc1d`) conforme a modelagem,
     possivelmente canal a canal.:contentReference[oaicite:2]{index=2}  

4. **Domínio da frequência**
   - Remover tendência residual (low‑freq),
   - enfatizar faixas ligadas a periodicidades de interesse
     (por exemplo, bandas associadas a “ciclos médios de swing”),
   - opcionalmente usar convoluções/correlações em frequência
     para imposição de “máscara” de Fibonacci ou de harmônicos.

5. **Transformada inversa**
   - Voltar ao domínio do tempo (`fftr1dinv`/`fftc1dinv`),
   - obter uma ou mais “waves” suavizadas por barra.

6. **Pós‑processamento**
   - Extrair:
     - valor da wave por barra,
     - magnitude/energia local,
     - possíveis pontos de reversão (cruzamentos, pontos 0 da wave, etc.),
     - um score de confiança por barra.

7. **Saída para o MT5**
   - A função exportada pela DLL deve encher buffers de saída:
     - `out_wave[]` (double por barra),
     - `out_confidence[]` (double 0..1 por barra),
     - opcionalmente flags discretas (int) para “provável topo/fundo”.

### 3.4 API esperada da DLL (exemplo conceitual)

Não é uma assinatura obrigatória, mas **guia** que o Codex deve seguir:

```c
// Chamado pelo indicador MQL5
extern "C" __declspec(dllexport)
int __stdcall gpu_reversal_wave_process(
    const double* price,     // [n]
    const double* volume,    // [n]
    const double* pivots,    // [n] codificação topo/fundo/none
    int           n,         // número de barras
    int           mode,      // flags para opções de pipeline
    double*       out_wave,        // [n] wave principal
    double*       out_confidence   // [n] score 0..1
    // se quiser: buffers extras de debug
);
Regras:

A função deve ser idempotente sobre a janela de dados recebida:

não manter estado interno que repinta de forma caótica,

se precisar de estado (ex.: incremental), expor outra função de reset
ou definir explicitamente a convenção.

### 3.4.1 Implementação atual na DLL

- O arquivo `native-fft/src/fasttransforms.cpp` agora exporta duas funções:
  - `gpu_reversal_wave_process(...)` – recebe `price[]`, `volume[]`, `pivots[]`,
    parâmetros (`window`, `modeFlags`, pesos) e devolve `wave[]`,
    `confidence[]` e `flags[]` já alinhados para o MT5.
  - `gpu_reversal_wave_synthetic_test(...)` – gera um cenário sintético e passa
    pelo mesmo pipeline, útil para smoke tests.
- Constantes de retorno (`RW_RESULT_*`) e `modeFlags` (`RW_MODE_*`) estão em
  `native-fft/include/fasttransforms.h` e espelhadas em
  `mql5/Include/ReversalWaveBridge.mqh`.
- O indicador `mql5/Indicators/ReversalWave.mq5` é apenas casca: coleta `close`,
  `volume` e pivôs via ZigZag, chama a DLL e desenha as waves.

MQL5 fica responsável apenas por:

preencher arrays com dados corretos (em séries temporais),

tratar a questão de “duas barras depois” para confirmação de pivô,

chamar esta função com n consistente.

3.5 Papel específico do agente REVERSAL_WAVE_GPU_PIPELINE

Quando você rodar o Codex com a tarefa correspondente, este agente deve:

Entender o contexto

Ler:

AGENTS.md (este arquivo),

native-fft/ (headers e .cpp),

o indicador MQL5 que será o consumidor da DLL,

qualquer arquivo TASK_REVERSAL_WAVE_GPU_PIPELINE.txt existente.

Projetar ou refinar a API da DLL

Garantir que a assinatura da DLL é estável e amigável
para MQL5 (tipos simples, double*, int, etc.).

Incluir opções para futura expansão (modes, parâmetros de tuning).

Implementar/ajustar o core C++/GPU

Implementar o pipeline descrito na seção 3.3,

garantir que todo o processamento acontece na DLL:

loops sobre barras,

FFT, convoluções, correlações, filtragens,

cálculo da wave e dos scores.

Enxugar o MQL5

Revisar .mq5 / .mqh:

remover qualquer lógica pesada,

garantir que o código só prepara dados, chama a DLL e plota.

Testar usando mtcli

Usar mtcli para:

compilar a DLL e o indicador,

instalar no MT5 de desenvolvimento do agente,

rodar smoke tests no Strategy Tester, se aplicável.

Documentar

Atualizar:

comentários no código C++,

comentários no indicador MQL5,

notas em AGENTS.md se a API da DLL mudar.
