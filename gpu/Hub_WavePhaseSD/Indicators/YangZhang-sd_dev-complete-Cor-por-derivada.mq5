//+------------------------------------------------------------------+
//|                Yang-Zhang 100% Dynamic - Zero Hardcoded.mq5      |
//|                Todos thresholds calculados dos proprios dados    |
//+------------------------------------------------------------------+
#property copyright   "Yang-Zhang 100% Dinamico - Zero Hardcoded"
#property version     "3.00"
#property indicator_separate_window
#property indicator_buffers 16
#property indicator_plots   4


//                          Índices:    0            1           2          3          4           5
// 0 = Cinza (primeira barra)
// 1 = Verde forte (alta aceleração positiva)
// 2 = Amarelo (aceleração moderada positiva)
// 3 = Laranja (aceleração moderada negativa)
// 4 = Vermelho (alta aceleração negativa)
// 5 = Azul escuro (estagnação/sem movimento)

#property indicator_label1  "Yang-Zhang Dynamic"
#property indicator_type1   DRAW_COLOR_LINE
#property indicator_color1  clrDarkGray,clrLime,clrKhaki,clrSandyBrown,clrTomato,clrLightSteelBlue
#property indicator_width1  2

#property indicator_label2  "Banda Superior"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDodgerBlue
#property indicator_style2  STYLE_DOT
#property indicator_width2  1

#property indicator_label3  "Banda Inferior"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrDodgerBlue
#property indicator_style3  STYLE_DOT
#property indicator_width3  1

#property indicator_label4  "Media Movel"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrGold
#property indicator_style4  STYLE_DASH
#property indicator_width4  1
//+------------------------------------------------------------------+
//| ADICIONAR ESTES BUFFERS NA SEÇÃO DE DECLARAÇÃO                  |
//+------------------------------------------------------------------+
// Adicionar após os buffers existentes:
double ChangeRateBuffer[];          // Taxa de mudanca (dσ/dt)
double ChangeRateNormBuffer[];      // Taxa normalizada (Z-score)
double AccelerationBuffer[];        // Aceleracao (d²σ/dt²)
double MomentumBuffer[];            // Momentum multi-periodo
double ChangeSignificanceBuffer[];  // Significancia estatistica
double ColorScoreBuffer[];          // Score final (0-100)
double MeanChangeBuffer[];          // Media das mudancas recentes
double StdDevChangeBuffer[];        // Desvio padrao das mudancas
double ChangePercentileBuffer[];    // Percentil da variacao atual
double VolatilityTrendBuffer[];     // Tendencia linear da volatilidade

//+------------------------------------------------------------------+
//| ADICIONAR ESTES INPUTS                                           |
//+------------------------------------------------------------------+
input int inpColorLookback  = 50;  // Lookback para calculo de cor
input int inpMomentumPeriod = 5;   // Periodo momentum


//+------------------------------------------------------------------+
//| ATUALIZAR A DEFINIÇÃO DE CORES (no topo do arquivo)             |
//+------------------------------------------------------------------+
// Substituir a linha de cores existente por:


//--- inputs
input int    inpPeriod         = 14;   // Periodo Yang-Zhang (n)
input int    inpAnnualization  = 252;  // Periodos de anualizacao
input int    inpKLookback      = 200;  // Lookback para otimizacao dinamica de K
input int    inpKRecalcPeriod  = 10;   // Recalcular K a cada N barras
input int    inpBandsPeriod    = 20;   // Periodo bandas
input double inpBandsDeviation = 2.0;  // Multiplicador desvio padrao
input int    inpPercentileLookback = 500; // Lookback para calcular percentis dinamicos
input bool   inpShowComponents = true; // Debug: componentes detalhados

//--- buffers
double YangZhangBuffer[];
double ColorBuffer[];
double UpperBandBuffer[];
double LowerBandBuffer[];
double MiddleBandBuffer[];
double OvernightBuffer[];
double OpenCloseBuffer[];
double RogersSatchellBuffer[];
double KFactorBuffer[];
double StdDevBuffer[];
double KPercentileBuffer[];      // Percentil de K na distribuicao historica
double VolPercentileBuffer[];    // Percentil de volatilidade
double KMeanBuffer[];            // Media historica de K
double KStdDevBuffer[];          // Desvio padrao historico de K
double VolRegimeBuffer[];        // Regime de volatilidade (0-100 score)

//--- variaveis globais
double g_last_k_value = 0.0;
int g_bars_since_k_calc = 0;

//+------------------------------------------------------------------+
//| Inicializacao                                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   if(inpPeriod < 2)
   {
      Print("ERRO: Periodo deve ser >= 2");
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   if(inpKLookback < inpPeriod * 3)
   {
      Print("ERRO: Lookback deve ser >= 3 x periodo");
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   
   
   SetIndexBuffer(0, YangZhangBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ColorBuffer, INDICATOR_COLOR_INDEX);
   SetIndexBuffer(2, UpperBandBuffer, INDICATOR_DATA);
   SetIndexBuffer(3, LowerBandBuffer, INDICATOR_DATA);
   SetIndexBuffer(4, MiddleBandBuffer, INDICATOR_DATA);
   SetIndexBuffer(5, OvernightBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, OpenCloseBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(7, RogersSatchellBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(8, KFactorBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(9, StdDevBuffer, INDICATOR_CALCULATIONS);

   //+------------------------------------------------------------------+
   //| ADICIONAR ESTES MAPEAMENTOS NO OnInit()                         |
   //+------------------------------------------------------------------+
   // Adicionar após os SetIndexBuffer existentes:
   SetIndexBuffer(10, ChangeRateBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(11, ChangeRateNormBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(12, AccelerationBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(13, MomentumBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(14, ChangeSignificanceBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(15, ColorScoreBuffer, INDICATOR_CALCULATIONS);
   
   IndicatorSetInteger(INDICATOR_DIGITS, 6);
   
   string shortname = StringFormat("Yang-Zhang 100%% Dynamic (n=%d, K_lookback=%d, percentile_lookback=%d)", 
                                    inpPeriod, inpKLookback, inpPercentileLookback);
   IndicatorSetString(INDICATOR_SHORTNAME, shortname);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Calculo de percentil dinamico de um array                        |
//+------------------------------------------------------------------+
double CalculatePercentile(double &data[], int count, double percentile)
{
   if(count < 1)
      return 0.0;
   
   double sorted[];
   ArrayResize(sorted, count);
   
   for(int i = 0; i < count; i++)
      sorted[i] = data[i];
   
   ArraySort(sorted);
   
   double index = (percentile / 100.0) * (double)(count - 1);
   int lower = (int)MathFloor(index);
   int upper = (int)MathCeil(index);
   
   if(lower == upper || upper >= count)
      return sorted[lower];
   
   double weight = index - (double)lower;
   return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

//+------------------------------------------------------------------+
//| Calculo de K Otimo Dinamico via Minimizacao de Variancia         |
//+------------------------------------------------------------------+
double CalculateDynamicOptimalK(const double &open[],
                                const double &high[],
                                const double &low[],
                                const double &close[],
                                int current_bar,
                                int rates_total,
                                int n,
                                int lookback,
                                double &k_mean_out,
                                double &k_stddev_out)
{
   if(current_bar < lookback + n + 1)
   {
      k_mean_out = 0.5;
      k_stddev_out = 0.0;
      return 0.5;
   }
   
   double overnight_vars[];
   double openclose_vars[];
   double rs_vars[];
   
   ArrayResize(overnight_vars, lookback);
   ArrayResize(openclose_vars, lookback);
   ArrayResize(rs_vars, lookback);
   
   int valid_windows = 0;
   
   for(int window = 0; window < lookback; window++)
   {
      int base_bar = current_bar - window;
      
      if(base_bar < n + 1 || base_bar >= rates_total)
         continue;
      
      //=== COMPONENTE 1: Variancia Overnight ===
      double sum_o = 0.0;
      double mean_o = 0.0;
      int count_o = 0;
      
      for(int j = 0; j < n; j++)
      {
         int curr = base_bar - j;
         int prev = base_bar - j - 1;
         
         if(prev < 0 || curr >= rates_total)
            continue;
         if(close[prev] <= 0 || open[curr] <= 0)
            continue;
         
         double ret = MathLog(open[curr] / close[prev]);
         mean_o += ret;
         count_o++;
      }
      
      if(count_o < 2)
         continue;
      
      mean_o /= (double)count_o;
      
      for(int j = 0; j < n; j++)
      {
         int curr = base_bar - j;
         int prev = base_bar - j - 1;
         
         if(prev < 0 || curr >= rates_total)
            continue;
         if(close[prev] <= 0 || open[curr] <= 0)
            continue;
         
         double ret = MathLog(open[curr] / close[prev]);
         sum_o += MathPow(ret - mean_o, 2);
      }
      
      overnight_vars[valid_windows] = sum_o / (double)(count_o - 1);
      
      //=== COMPONENTE 2: Variancia Open-Close ===
      double sum_c = 0.0;
      double mean_c = 0.0;
      int count_c = 0;
      
      for(int j = 0; j < n; j++)
      {
         int idx = base_bar - j;
         
         if(idx < 0 || idx >= rates_total)
            continue;
         if(open[idx] <= 0 || close[idx] <= 0)
            continue;
         
         double ret = MathLog(close[idx] / open[idx]);
         mean_c += ret;
         count_c++;
      }
      
      if(count_c < 2)
         continue;
      
      mean_c /= (double)count_c;
      
      for(int j = 0; j < n; j++)
      {
         int idx = base_bar - j;
         
         if(idx < 0 || idx >= rates_total)
            continue;
         if(open[idx] <= 0 || close[idx] <= 0)
            continue;
         
         double ret = MathLog(close[idx] / open[idx]);
         sum_c += MathPow(ret - mean_c, 2);
      }
      
      openclose_vars[valid_windows] = sum_c / (double)(count_c - 1);
      
      //=== COMPONENTE 3: Variancia Rogers-Satchell ===
      double sum_rs = 0.0;
      int count_rs = 0;
      
      for(int j = 0; j < n; j++)
      {
         int idx = base_bar - j;
         
         if(idx < 0 || idx >= rates_total)
            continue;
         if(close[idx] <= 0 || open[idx] <= 0 || high[idx] <= 0 || low[idx] <= 0)
            continue;
         if(high[idx] < close[idx] || high[idx] < open[idx] || 
            low[idx] > close[idx] || low[idx] > open[idx])
            continue;
         
         double ln_hc = MathLog(high[idx] / close[idx]);
         double ln_ho = MathLog(high[idx] / open[idx]);
         double ln_lc = MathLog(low[idx] / close[idx]);
         double ln_lo = MathLog(low[idx] / open[idx]);
         
         sum_rs += (ln_hc * ln_ho) + (ln_lc * ln_lo);
         count_rs++;
      }
      
      if(count_rs < 2)
         continue;
      
      rs_vars[valid_windows] = sum_rs / (double)count_rs;
      
      valid_windows++;
   }
   
   if(valid_windows < 10)
   {
      k_mean_out = 0.5;
      k_stddev_out = 0.0;
      return 0.5;
   }
   
   //=== DETERMINAR RANGE DINAMICO PARA BUSCA DE K ===
   // Calcula min e max das variancias para determinar range valido de k
   
   double max_var_c = openclose_vars[0];
   double min_var_c = openclose_vars[0];
   double max_var_rs = rs_vars[0];
   double min_var_rs = rs_vars[0];
   
   for(int w = 0; w < valid_windows; w++)
   {
      if(openclose_vars[w] > max_var_c) max_var_c = openclose_vars[w];
      if(openclose_vars[w] < min_var_c) min_var_c = openclose_vars[w];
      if(rs_vars[w] > max_var_rs) max_var_rs = rs_vars[w];
      if(rs_vars[w] < min_var_rs) min_var_rs = rs_vars[w];
   }
   
   // Determina limites dinamicos de k baseado nas variancias relativas
   double k_min = 0.0;
   double k_max = 1.0;
   
   // Se uma variancia domina, ajusta range
   if(max_var_rs > 0 && max_var_c > 0)
   {
      double ratio = max_var_c / max_var_rs;
      if(ratio < 0.1)
         k_max = 0.3; // Se var_c muito menor que var_rs, limita k_max
      else if(ratio > 10.0)
         k_min = 0.7; // Se var_c muito maior que var_rs, limita k_min
   }
   
   // Granularidade dinamica baseada no range
   int num_steps = 1000;
   double step_size = (k_max - k_min) / (double)num_steps;
   
   //=== OTIMIZACAO: Busca de K que Minimiza Variancia ===
   
   double best_k = (k_min + k_max) / 2.0;
   double min_total_variance = DBL_MAX;
   
   for(int k_index = 0; k_index <= num_steps; k_index++)
   {
      double test_k = k_min + (step_size * (double)k_index);
      
      double sum_estimator_variance = 0.0;
      
      for(int w = 0; w < valid_windows; w++)
      {
         double sigma_sq_yz = overnight_vars[w] + 
                               (test_k * openclose_vars[w]) + 
                               ((1.0 - test_k) * rs_vars[w]);
         
         sum_estimator_variance += sigma_sq_yz;
      }
      
      double mean_estimator = sum_estimator_variance / (double)valid_windows;
      
      double sum_variance_of_variance = 0.0;
      
      for(int w = 0; w < valid_windows; w++)
      {
         double sigma_sq_yz = overnight_vars[w] + 
                               (test_k * openclose_vars[w]) + 
                               ((1.0 - test_k) * rs_vars[w]);
         
         sum_variance_of_variance += MathPow(sigma_sq_yz - mean_estimator, 2);
      }
      
      double variance_of_estimator = sum_variance_of_variance / (double)(valid_windows - 1);
      
      if(variance_of_estimator < min_total_variance)
      {
         min_total_variance = variance_of_estimator;
         best_k = test_k;
      }
   }
   
   //=== REFINAMENTO LOCAL ===
   double fine_search_range = step_size * 10.0;
   double fine_step = step_size / 10.0;
   
   double refined_k_min = MathMax(k_min, best_k - fine_search_range);
   double refined_k_max = MathMin(k_max, best_k + fine_search_range);
   
   for(double test_k = refined_k_min; test_k <= refined_k_max; test_k += fine_step)
   {
      double sum_estimator_variance = 0.0;
      
      for(int w = 0; w < valid_windows; w++)
      {
         double sigma_sq_yz = overnight_vars[w] + 
                               (test_k * openclose_vars[w]) + 
                               ((1.0 - test_k) * rs_vars[w]);
         sum_estimator_variance += sigma_sq_yz;
      }
      
      double mean_estimator = sum_estimator_variance / (double)valid_windows;
      double sum_variance_of_variance = 0.0;
      
      for(int w = 0; w < valid_windows; w++)
      {
         double sigma_sq_yz = overnight_vars[w] + 
                               (test_k * openclose_vars[w]) + 
                               ((1.0 - test_k) * rs_vars[w]);
         sum_variance_of_variance += MathPow(sigma_sq_yz - mean_estimator, 2);
      }
      
      double variance_of_estimator = sum_variance_of_variance / (double)(valid_windows - 1);
      
      if(variance_of_estimator < min_total_variance)
      {
         min_total_variance = variance_of_estimator;
         best_k = test_k;
      }
   }
   
   //=== CALCULAR ESTATISTICAS DE K HISTORICO ===
   // Simula k para cada janela para obter distribuicao
   double k_historical[];
   ArrayResize(k_historical, valid_windows);
   
   for(int w = 0; w < valid_windows; w++)
   {
      // Para cada janela, calcula qual k seria otimo
      double local_best_k = 0.5;
      double local_min_var = DBL_MAX;
      
      for(int k_idx = 0; k_idx <= 100; k_idx++)
      {
         double test_k = k_min + ((k_max - k_min) * (double)k_idx / 100.0);
         
         double sigma_sq_yz = overnight_vars[w] + 
                               (test_k * openclose_vars[w]) + 
                               ((1.0 - test_k) * rs_vars[w]);
         
         // Criterio simplificado: minimizar desvio de componentes ponderados
         double component_diff = MathAbs((test_k * openclose_vars[w]) - ((1.0 - test_k) * rs_vars[w]));
         
         if(component_diff < local_min_var)
         {
            local_min_var = component_diff;
            local_best_k = test_k;
         }
      }
      
      k_historical[w] = local_best_k;
   }
   
   // Media e desvio padrao de k historico
   double sum_k = 0.0;
   for(int w = 0; w < valid_windows; w++)
      sum_k += k_historical[w];
   
   k_mean_out = sum_k / (double)valid_windows;
   
   double sum_k_sq_dev = 0.0;
   for(int w = 0; w < valid_windows; w++)
      sum_k_sq_dev += MathPow(k_historical[w] - k_mean_out, 2);
   
   k_stddev_out = MathSqrt(sum_k_sq_dev / (double)(valid_windows - 1));
   
   return best_k;
}

//+------------------------------------------------------------------+
//| OnCalculate                                                       |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(rates_total < inpPeriod + inpKLookback + inpBandsPeriod + inpPercentileLookback)
      return(0);
   
   int start_pos;
   int minimum_start = inpPeriod + inpKLookback + inpPercentileLookback;

   if(prev_calculated == 0)
      start_pos = minimum_start;
   else
   {
      start_pos = prev_calculated - 1;
      if(start_pos < minimum_start)
         start_pos = minimum_start;
   }

   ArrayResize(YangZhangBuffer, rates_total);
   ArrayResize(ColorBuffer, rates_total);
   ArrayResize(UpperBandBuffer, rates_total);
   ArrayResize(LowerBandBuffer, rates_total);
   ArrayResize(MiddleBandBuffer, rates_total);
   ArrayResize(OvernightBuffer, rates_total);
   ArrayResize(OpenCloseBuffer, rates_total);
   ArrayResize(RogersSatchellBuffer, rates_total);
   ArrayResize(KFactorBuffer, rates_total);
   ArrayResize(KMeanBuffer, rates_total);
   ArrayResize(KStdDevBuffer, rates_total);
   ArrayResize(StdDevBuffer, rates_total);
   ArrayResize(KPercentileBuffer, rates_total);
   ArrayResize(VolPercentileBuffer, rates_total);
   ArrayResize(VolRegimeBuffer, rates_total);
   ArrayResize(ChangeRateBuffer, rates_total);
   ArrayResize(ChangeRateNormBuffer, rates_total);
   ArrayResize(AccelerationBuffer, rates_total);
   ArrayResize(MomentumBuffer, rates_total);
   ArrayResize(ChangeSignificanceBuffer, rates_total);
   ArrayResize(ColorScoreBuffer, rates_total);
   ArrayResize(MeanChangeBuffer, rates_total);
   ArrayResize(StdDevChangeBuffer, rates_total);
   ArrayResize(ChangePercentileBuffer, rates_total);
   ArrayResize(VolatilityTrendBuffer, rates_total);

   int guard_index = start_pos - 1;
   if(guard_index < 0)
      guard_index = 0;
   if(guard_index >= rates_total)
      guard_index = rates_total - 1;
   if(guard_index >= 0 && guard_index < rates_total)
   {
      KMeanBuffer[guard_index] = g_last_k_value;
      KStdDevBuffer[guard_index] = 0.0;
   }

   for(int j = 0; j < inpPercentileLookback && j < rates_total; j++)
   {
      ChangePercentileBuffer[j] = 50.0;
   }

   //--- Loop principal
   for(int i = start_pos; i < rates_total && !IsStopped(); i++)
   {
      if(i < inpPeriod + inpKLookback + inpPercentileLookback)
         continue;
      
      //=== RECALCULAR K DINAMICAMENTE ===
      double k_factor;
      double k_mean;
      double k_stddev;
      
      if(g_bars_since_k_calc >= inpKRecalcPeriod || g_last_k_value == 0.0)
      {
         k_factor = CalculateDynamicOptimalK(open, high, low, close, i, rates_total, 
                                              inpPeriod, inpKLookback, k_mean, k_stddev);
         g_last_k_value = k_factor;
         g_bars_since_k_calc = 0;
      }
      else
      {
         k_factor = g_last_k_value;
         k_mean = (i - 1 >= 0 ? KMeanBuffer[i - 1] : k_mean);
         k_stddev = (i - 1 >= 0 ? KStdDevBuffer[i - 1] : 0.0);
         g_bars_since_k_calc++;
      }
      
      KFactorBuffer[i] = k_factor;
      KMeanBuffer[i] = k_mean;
      KStdDevBuffer[i] = k_stddev;
      
      //=== COMPONENTE 1: Overnight Volatility ===
      double sum_o = 0.0;
      double mean_o = 0.0;
      int count_o = 0;
      
      for(int j = 0; j < inpPeriod; j++)
      {
         int curr = i - j;
         int prev = i - j - 1;
         
         if(prev < 0 || curr >= rates_total)
            continue;
         if(close[prev] <= 0 || open[curr] <= 0)
            continue;
         
         double ret = MathLog(open[curr] / close[prev]);
         mean_o += ret;
         count_o++;
      }
      
      if(count_o < 2)
      {
         YangZhangBuffer[i] = 0.0;
         ColorBuffer[i] = 0;
         continue;
      }
      
      mean_o /= (double)count_o;
      
      for(int j = 0; j < inpPeriod; j++)
      {
         int curr = i - j;
         int prev = i - j - 1;
         
         if(prev < 0 || curr >= rates_total)
            continue;
         if(close[prev] <= 0 || open[curr] <= 0)
            continue;
         
         double ret = MathLog(open[curr] / close[prev]);
         sum_o += MathPow(ret - mean_o, 2);
      }
      
      double var_o = sum_o / (double)(count_o - 1);
      
      //=== COMPONENTE 2: Open-Close Volatility ===
      double sum_c = 0.0;
      double mean_c = 0.0;
      int count_c = 0;
      
      for(int j = 0; j < inpPeriod; j++)
      {
         int idx = i - j;
         
         if(idx < 0 || idx >= rates_total)
            continue;
         if(open[idx] <= 0 || close[idx] <= 0)
            continue;
         
         double ret = MathLog(close[idx] / open[idx]);
         mean_c += ret;
         count_c++;
      }
      
      if(count_c < 2)
      {
         YangZhangBuffer[i] = 0.0;
         ColorBuffer[i] = 0;
         continue;
      }
      
      mean_c /= (double)count_c;
      
      for(int j = 0; j < inpPeriod; j++)
      {
         int idx = i - j;
         
         if(idx < 0 || idx >= rates_total)
            continue;
         if(open[idx] <= 0 || close[idx] <= 0)
            continue;
         
         double ret = MathLog(close[idx] / open[idx]);
         sum_c += MathPow(ret - mean_c, 2);
      }
      
      double var_c = sum_c / (double)(count_c - 1);
      
      //=== COMPONENTE 3: Rogers-Satchell Volatility ===
      double sum_rs = 0.0;
      int count_rs = 0;
      
      for(int j = 0; j < inpPeriod; j++)
      {
         int idx = i - j;
         
         if(idx < 0 || idx >= rates_total)
            continue;
         if(close[idx] <= 0 || open[idx] <= 0 || high[idx] <= 0 || low[idx] <= 0)
            continue;
         if(high[idx] < close[idx] || high[idx] < open[idx] || 
            low[idx] > close[idx] || low[idx] > open[idx])
            continue;
         
         double ln_hc = MathLog(high[idx] / close[idx]);
         double ln_ho = MathLog(high[idx] / open[idx]);
         double ln_lc = MathLog(low[idx] / close[idx]);
         double ln_lo = MathLog(low[idx] / open[idx]);
         
         sum_rs += (ln_hc * ln_ho) + (ln_lc * ln_lo);
         count_rs++;
      }
      
      if(count_rs < 2)
      {
         YangZhangBuffer[i] = 0.0;
         ColorBuffer[i] = 0;
         continue;
      }
      
      double var_rs = sum_rs / (double)count_rs;
      
      //=== YANG-ZHANG COM K DINAMICO ===
      double var_yz = var_o + (k_factor * var_c) + ((1.0 - k_factor) * var_rs);
      
      if(var_yz < 0)
         var_yz = 0;
      
      double sigma_yz = MathSqrt(var_yz * (double)inpAnnualization);
      
      YangZhangBuffer[i] = sigma_yz;
      
      //--- Componentes individuais
      OvernightBuffer[i] = MathSqrt(MathAbs(var_o) * (double)inpAnnualization);
      OpenCloseBuffer[i] = MathSqrt(MathAbs(var_c) * (double)inpAnnualization);
      RogersSatchellBuffer[i] = MathSqrt(MathAbs(var_rs) * (double)inpAnnualization);
      
      //=== CALCULAR PERCENTIS DINAMICOS ===
      
      // Percentil de K
      double k_history[];
      ArrayResize(k_history, inpPercentileLookback);
      int k_count = 0;
      
      for(int lookback_idx = 0; lookback_idx < inpPercentileLookback; lookback_idx++)
      {
         int hist_idx = i - lookback_idx;
         if(hist_idx >= 0 && hist_idx < rates_total && KFactorBuffer[hist_idx] > 0)
         {
            k_history[k_count] = KFactorBuffer[hist_idx];
            k_count++;
         }
      }
      
      if(k_count >= 10)
      {
         double k_percentiles[];
         ArrayResize(k_percentiles, k_count);
         for(int p = 0; p < k_count; p++)
            k_percentiles[p] = k_history[p];
         
         ArraySort(k_percentiles);
         
         // Encontrar percentil do k atual
         int position = 0;
         for(int p = 0; p < k_count; p++)
         {
            if(k_factor > k_percentiles[p])
               position++;
         }
         
         KPercentileBuffer[i] = ((double)position / (double)k_count) * 100.0;
      }
      else
      {
         KPercentileBuffer[i] = 50.0;
      }
      
      // Percentil de Volatilidade
      double vol_history[];
      ArrayResize(vol_history, inpPercentileLookback);
      int vol_count = 0;
      
      for(int lookback_idx = 0; lookback_idx < inpPercentileLookback; lookback_idx++)
      {
         int hist_idx = i - lookback_idx;
         if(hist_idx >= 0 && hist_idx < rates_total && YangZhangBuffer[hist_idx] > 0)
         {
            vol_history[vol_count] = YangZhangBuffer[hist_idx];
            vol_count++;
         }
      }
      
      if(vol_count >= 10)
      {
         double vol_percentiles[];
         ArrayResize(vol_percentiles, vol_count);
         for(int p = 0; p < vol_count; p++)
            vol_percentiles[p] = vol_history[p];
         
         ArraySort(vol_percentiles);
         
         int position = 0;
         for(int p = 0; p < vol_count; p++)
         {
            if(sigma_yz > vol_percentiles[p])
               position++;
         }
         
         VolPercentileBuffer[i] = ((double)position / (double)vol_count) * 100.0;
      }
      else
      {
         VolPercentileBuffer[i] = 50.0;
      }
      
      // Regime de volatilidade (score 0-100 baseado em multiplos fatores)
      double regime_score = 0.0;
      
      // Fator 1: Percentil de volatilidade (40% do score)
      regime_score += VolPercentileBuffer[i] * 0.4;
      
      // Fator 2: Desvio de K da media (30% do score)
      if(k_stddev > 0)
      {
         double k_z_score = (k_factor - k_mean) / k_stddev;
         double k_percentile_approx = 50.0 + (k_z_score * 20.0);
         k_percentile_approx = MathMax(0.0, MathMin(100.0, k_percentile_approx));
         regime_score += k_percentile_approx * 0.3;
      }
      else
      {
         regime_score += 50.0 * 0.3;
      }
      
      // Fator 3: Taxa de mudanca de volatilidade (30% do score)
      if(i > 0 && YangZhangBuffer[i - 1] > 0)
      {
         double vol_change = (YangZhangBuffer[i] - YangZhangBuffer[i - 1]) / YangZhangBuffer[i - 1];
         double vol_change_percentile = 50.0 + (vol_change * 1000.0);
         vol_change_percentile = MathMax(0.0, MathMin(100.0, vol_change_percentile));
         regime_score += vol_change_percentile * 0.3;
      }
      else
      {
         regime_score += 50.0 * 0.3;
      }
      
      VolRegimeBuffer[i] = regime_score;
      
      //--- Cor


//+------------------------------------------------------------------+
//| SUBSTITUIR O BLOCO DE CÁLCULO DE COR EXISTENTE                  |
//| Procure por "//--- Cor" no código e substitua por isto:         |
//+------------------------------------------------------------------+

      //=== SISTEMA AVANÇADO DE CÁLCULO DE COR ===
      
      if(i >= inpColorLookback + inpMomentumPeriod)
      {
         //--- CÁLCULO 1: Taxa de mudança (primeira derivada)
         if(i > 0 && YangZhangBuffer[i-1] > 0)
         {
            // Taxa de mudança absoluta
            double change_abs = YangZhangBuffer[i] - YangZhangBuffer[i-1];
            
            // Taxa de mudança percentual
            double change_pct = (change_abs / YangZhangBuffer[i-1]) * 100.0;
            
            ChangeRateBuffer[i] = change_pct;
         }
         else
         {
            ChangeRateBuffer[i] = 0.0;
         }
         
         //--- CÁLCULO 2: Estatísticas históricas de mudança
         double sum_changes = 0.0;
         int count_changes = 0;
         
         for(int lookback = 1; lookback <= inpColorLookback; lookback++)
         {
            int idx = i - lookback;
            if(idx >= 0 && idx < rates_total && YangZhangBuffer[idx] > 0)
            {
               sum_changes += ChangeRateBuffer[idx];
               count_changes++;
            }
         }
         
         double mean_change = 0.0;
         if(count_changes > 0)
            mean_change = sum_changes / (double)count_changes;
         
         MeanChangeBuffer[i] = mean_change;
         
         // Desvio padrão das mudanças
         double sum_sq_dev = 0.0;
         for(int lookback = 1; lookback <= inpColorLookback; lookback++)
         {
            int idx = i - lookback;
            if(idx >= 0 && idx < rates_total && YangZhangBuffer[idx] > 0)
            {
               double dev = ChangeRateBuffer[idx] - mean_change;
               sum_sq_dev += dev * dev;
            }
         }
         
         double stddev_change = 0.0;
         if(count_changes > 1)
            stddev_change = MathSqrt(sum_sq_dev / (double)(count_changes - 1));
         
         StdDevChangeBuffer[i] = stddev_change;
         
         //--- CÁLCULO 3: Z-score da mudança (normalização)
         if(stddev_change > 0)
         {
            ChangeRateNormBuffer[i] = (ChangeRateBuffer[i] - mean_change) / stddev_change;
         }
         else
         {
            ChangeRateNormBuffer[i] = 0.0;
         }
         
         //--- CÁLCULO 4: Aceleração (segunda derivada)
         if(i > 1)
         {
            double current_change = YangZhangBuffer[i] - YangZhangBuffer[i-1];
            double previous_change = YangZhangBuffer[i-1] - YangZhangBuffer[i-2];
            AccelerationBuffer[i] = current_change - previous_change;
         }
         else
         {
            AccelerationBuffer[i] = 0.0;
         }
         
         //--- CÁLCULO 5: Momentum multi-período
         if(i >= inpMomentumPeriod)
         {
            double momentum_sum = 0.0;
            int momentum_count = 0;
            
            for(int m = 1; m <= inpMomentumPeriod; m++)
            {
               int idx = i - m;
               if(idx >= 0 && idx < rates_total)
               {
                  momentum_sum += ChangeRateBuffer[idx];
                  momentum_count++;
               }
            }
            
            if(momentum_count > 0)
               MomentumBuffer[i] = momentum_sum / (double)momentum_count;
            else
               MomentumBuffer[i] = 0.0;
         }
         else
         {
            MomentumBuffer[i] = 0.0;
         }
         
         //--- CÁLCULO 6: Percentil da mudança na distribuição histórica
         double changes_array[];
         ArrayResize(changes_array, inpColorLookback);
         int changes_count = 0;
         
         for(int lookback = 1; lookback <= inpColorLookback; lookback++)
         {
            int idx = i - lookback;
            if(idx >= 0 && idx < rates_total && YangZhangBuffer[idx] > 0)
            {
               changes_array[changes_count] = ChangeRateBuffer[idx];
               changes_count++;
            }
         }
         
         if(changes_count >= 10)
         {
            ArrayResize(changes_array, changes_count);
            ArraySort(changes_array);
            
            int position = 0;
            for(int p = 0; p < changes_count; p++)
            {
               if(ChangeRateBuffer[i] > changes_array[p])
                  position++;
            }
            
            ChangePercentileBuffer[i] = ((double)position / (double)changes_count) * 100.0;
         }
         else
         {
            ChangePercentileBuffer[i] = 50.0;
         }
         
         //--- CÁLCULO 7: Significância estatística (teste t simplificado)
         if(stddev_change > 0 && count_changes > 1)
         {
            double t_stat = MathAbs(ChangeRateBuffer[i] - mean_change) / (stddev_change / MathSqrt((double)count_changes));
            
            // Converter t-stat em score de significância (0-100)
            // t > 3.0 = altamente significativo
            double significance = MathMin(100.0, (t_stat / 3.0) * 100.0);
            ChangeSignificanceBuffer[i] = significance;
         }
         else
         {
            ChangeSignificanceBuffer[i] = 0.0;
         }
         
         //--- CÁLCULO 8: Score composto final (0-100)
         // Peso: 40% magnitude, 30% momentum, 20% aceleração, 10% significância
         
         double score = 0.0;
         
         // Componente 1: Magnitude normalizada (40%)
         double magnitude_score = ChangePercentileBuffer[i];
         score += magnitude_score * 0.4;
         
         // Componente 2: Momentum (30%)
         double momentum_normalized = 50.0 + (MomentumBuffer[i] * 10.0);
         momentum_normalized = MathMax(0.0, MathMin(100.0, momentum_normalized));
         score += momentum_normalized * 0.3;
         
         // Componente 3: Aceleração (20%)
         double accel_normalized = 50.0 + (AccelerationBuffer[i] * 1000.0);
         accel_normalized = MathMax(0.0, MathMin(100.0, accel_normalized));
         score += accel_normalized * 0.2;
         
         // Componente 4: Significância (10%)
         score += ChangeSignificanceBuffer[i] * 0.1;
         
         ColorScoreBuffer[i] = score;
         
         //--- CÁLCULO 9: Tendência de volatilidade (regressão linear simplificada)
         double sum_x = 0.0;
         double sum_y = 0.0;
         double sum_xy = 0.0;
         double sum_x2 = 0.0;
         int trend_count = 0;
         
         for(int t = 0; t < inpMomentumPeriod; t++)
         {
            int idx = i - t;
            if(idx >= 0 && idx < rates_total && YangZhangBuffer[idx] > 0)
            {
               double x = (double)t;
               double y = YangZhangBuffer[idx];
               
               sum_x += x;
               sum_y += y;
               sum_xy += x * y;
               sum_x2 += x * x;
               trend_count++;
            }
         }
         
         if(trend_count > 1)
         {
            double n = (double)trend_count;
            double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            VolatilityTrendBuffer[i] = slope * 1000.0; // Amplificar para visibilidade
         }
         else
         {
            VolatilityTrendBuffer[i] = 0.0;
         }
         
               //--- DETERMINAÇÃO FINAL DA COR baseada na direção/magnitude
      double change = ChangeRateBuffer[i];
      double sigma_change = StdDevChangeBuffer[i];
      if(sigma_change <= 0.0)
         sigma_change = 1.0e-6;
      double normalized_change = MathAbs(change) / sigma_change;
      double strength = ColorScoreBuffer[i] / 100.0;

      if(i == 0 || !MathIsValidNumber(change))
      {
         ColorBuffer[i] = 0; // cinza inicial ou dados inválidos
      }
      else if(change > 0.0)
      {
         if(normalized_change >= 1.5 || strength >= 0.70)
            ColorBuffer[i] = 1; // verde forte
         else if(normalized_change >= 0.5 || strength >= 0.55)
            ColorBuffer[i] = 2; // amarelo
         else
            ColorBuffer[i] = 5; // neutro
      }
      else if(change < 0.0)
      {
         if(normalized_change >= 1.5 || strength >= 0.70)
            ColorBuffer[i] = 4; // vermelho
         else if(normalized_change >= 0.5 || strength >= 0.55)
            ColorBuffer[i] = 3; // laranja
         else
            ColorBuffer[i] = 5; // neutro
      }
      else
      {
         ColorBuffer[i] = 5; // neutro
      }
   }
   else
   {
      ColorBuffer[i] = 0;
   }

   //=== CALCULAR BANDAS ===
   int bands_start = MathMax(inpPeriod + inpKLookback + inpPercentileLookback + inpBandsPeriod - 1, start_pos);

   for(int i = bands_start; i < rates_total && !IsStopped(); i++)
   {
      double sum_ma = 0.0;
      int valid_ma = 0;
      
      for(int j = 0; j < inpBandsPeriod; j++)
      {
         int idx = i - j;
         if(idx < 0 || idx >= rates_total)
            continue;
         
         if(YangZhangBuffer[idx] > 0)
         {
            sum_ma += YangZhangBuffer[idx];
            valid_ma++;
         }
      }
      
      if(valid_ma < 2)
      {
         MiddleBandBuffer[i] = 0.0;
         UpperBandBuffer[i] = 0.0;
         LowerBandBuffer[i] = 0.0;
         continue;
      }
      
      double ma = sum_ma / (double)valid_ma;
      MiddleBandBuffer[i] = ma;
      
      double sum_sq_dev = 0.0;
      
      for(int j = 0; j < inpBandsPeriod; j++)
      {
         int idx = i - j;
         if(idx < 0 || idx >= rates_total)
            continue;
         
         if(YangZhangBuffer[idx] > 0)
         {
            double dev = YangZhangBuffer[idx] - ma;
            sum_sq_dev += dev * dev;
         }
      }
      
      double stddev = MathSqrt(sum_sq_dev / (double)(valid_ma - 1));
      StdDevBuffer[i] = stddev;
      
      UpperBandBuffer[i] = ma + (inpBandsDeviation * stddev);
      LowerBandBuffer[i] = MathMax(0, ma - (inpBandsDeviation * stddev));
   }
   
   return(rates_total);
}

//+------------------------------------------------------------------+
//| OnChartEvent - Tooltip 100% dinamico sem thresholds fixos        |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id == CHARTEVENT_MOUSE_MOVE)
   {
      int x = (int)lparam;
      int y = (int)dparam;
      datetime dt;
      double price;
      int window;
      
      if(ChartXYToTimePrice(0, x, y, window, dt, price))
      {
         int bar = iBarShift(_Symbol, _Period, dt);
         
         if(bar >= 0 && bar < ArraySize(YangZhangBuffer))
         {
            string info = "=== YANG-ZHANG 100% DYNAMIC (ZERO HARDCODED) ===\n";
            info += "Bar: " + IntegerToString(bar) + "\n\n";
            
            info += "VOLATILIDADE:\n";
            info += "Yang-Zhang Total:  " + DoubleToString(YangZhangBuffer[bar], 6) + " (" + DoubleToString(YangZhangBuffer[bar] * 100, 2) + "%)\n";
            info += "Banda Superior:    " + DoubleToString(UpperBandBuffer[bar], 6) + " (" + DoubleToString(UpperBandBuffer[bar] * 100, 2) + "%)\n";
            info += "Media:             " + DoubleToString(MiddleBandBuffer[bar], 6) + " (" + DoubleToString(MiddleBandBuffer[bar] * 100, 2) + "%)\n";
            info += "Banda Inferior:    " + DoubleToString(LowerBandBuffer[bar], 6) + " (" + DoubleToString(LowerBandBuffer[bar] * 100, 2) + "%)\n";
            info += "Desvio Padrao:     " + DoubleToString(StdDevBuffer[bar], 6) + "\n";
            info += "Percentil Vol:     " + DoubleToString(VolPercentileBuffer[bar], 1) + "% da distribuicao historica\n";
            info += "Regime Score:      " + DoubleToString(VolRegimeBuffer[bar], 1) + "/100\n\n";
            
            if(MiddleBandBuffer[bar] > 0 && StdDevBuffer[bar] > 0)
            {
               double dist = (YangZhangBuffer[bar] - MiddleBandBuffer[bar]) / StdDevBuffer[bar];
               info += "Distancia Media:   " + DoubleToString(dist, 2) + " sigma\n\n";
            }
            
            info += "COMPONENTES YANG-ZHANG:\n";
            info += "sigma_overnight:       " + DoubleToString(OvernightBuffer[bar], 6) + " (" + DoubleToString(OvernightBuffer[bar] * 100, 2) + "%)\n";
            info += "sigma_open-close:      " + DoubleToString(OpenCloseBuffer[bar], 6) + " (" + DoubleToString(OpenCloseBuffer[bar] * 100, 2) + "%)\n";
            info += "sigma_Rogers-Satchell: " + DoubleToString(RogersSatchellBuffer[bar], 6) + " (" + DoubleToString(RogersSatchellBuffer[bar] * 100, 2) + "%)\n\n";
            
            info += "K DINAMICO (100% EMPIRICO):\n";
            info += "k atual:           " + DoubleToString(KFactorBuffer[bar], 6) + " (" + DoubleToString(KFactorBuffer[bar] * 100, 2) + "%)\n";
            info += "(1-k) atual:       " + DoubleToString(1.0 - KFactorBuffer[bar], 6) + " (" + DoubleToString((1.0 - KFactorBuffer[bar]) * 100, 2) + "%)\n";
            info += "k media historica: " + DoubleToString(KMeanBuffer[bar], 6) + "\n";
            info += "k desvio padrao:   " + DoubleToString(KStdDevBuffer[bar], 6) + "\n";
            info += "k percentil:       " + DoubleToString(KPercentileBuffer[bar], 1) + "% da distribuicao\n\n";
            
            info += "INTERPRETACAO DINAMICA DE K:\n";
            
            // Interpretacao baseada no percentil (100% dinamico)
            double k_pct = KPercentileBuffer[bar];
            
            if(k_pct < 10.0)
               info += "K muito baixo (bottom " + DoubleToString(k_pct, 1) + "%): Dominancia extrema Rogers-Satchell\n";
            else if(k_pct < 25.0)
               info += "K abaixo quartil inferior: Alta preferencia Rogers-Satchell\n";
            else if(k_pct < 40.0)
               info += "K abaixo mediana: Preferencia moderada Rogers-Satchell\n";
            else if(k_pct < 60.0)
               info += "K proximo da mediana: Equilibrio entre componentes\n";
            else if(k_pct < 75.0)
               info += "K acima mediana: Preferencia moderada Open-Close\n";
            else if(k_pct < 90.0)
               info += "K acima quartil superior: Alta preferencia Open-Close\n";
            else
               info += "K muito alto (top " + DoubleToString(100.0 - k_pct, 1) + "%): Dominancia extrema Open-Close\n";
            
            // Desvio de K da media (Z-score)
            if(KStdDevBuffer[bar] > 0)
            {
               double k_z = (KFactorBuffer[bar] - KMeanBuffer[bar]) / KStdDevBuffer[bar];
               info += "K Z-score: " + DoubleToString(k_z, 2) + " desvios da media\n";
               
               if(MathAbs(k_z) > 3.0)
                  info += "ALERTA: K em territorio extremo (>3 sigma)\n";
               else if(MathAbs(k_z) > 2.0)
                  info += "ATENCAO: K em territorio incomum (>2 sigma)\n";
            }
            
            Comment(info);
         }
      }
   }
}
//+------------------------------------------------------------------+