#property copyright "Reversal Wave GPU Pipeline"
#property link      "https://github.com/sindlinger/metacli"
#property version   "1.00"
#property strict
#property description "Wrapper indicator that marshals data to the GPU reversal wave DLL."
#property indicator_separate_window
#property indicator_plots 4
#property indicator_buffers 8  // 4 plot buffers + 4 cálculos internos (V2 waves)

#define RW_RESULT_OK                0
#define RW_RESULT_INVALID_ARGUMENT -1
#define RW_RESULT_NOT_ENOUGH_DATA  -2

#define RW_MODE_HIGHPASS        1
#define RW_MODE_EMPHASIZE_PIVOT 2
#define RW_MODE_USE_HANN        4
#define RW_MODE_EXTENDED_BAND   8

#define RW_FLAG_BULLISH         1
#define RW_FLAG_BEARISH         2
#define RW_FLAG_LOW_CONFIDENCE  4
#define RW_FLAG_WARMUP          8

#import "fasttransforms.dll"
int gpu_reversal_wave_process(const double &price[],
                              const double &volume[],
                              const double &pivots[],
                              int length,
                              int window,
                              int modeFlags,
                              double priceWeight,
                              double volumeWeight,
                              double pivotWeight,
                              double &outWave[],
                              double &outConfidence[],
                              int &outFlags[]);

int gpu_reversal_wave_synthetic_test(int length,
                                     double oscillation,
                                     double noiseLevel,
                                     double &outWave[],
                                     double &outConfidence[],
                                     int &outFlags[]);

int gpu_reversal_wave_process_v2(const double &open[],
                                 const double &price[],
                                 const double &high[],
                                 const double &low[],
                                 const double &volume[],
                                 const double &pivots[],
                                 int length,
                                 int window,
                                 int modeFlags,
                                 double priceWeight,
                                 double volumeWeight,
                                 double pivotWeight,
                                 double candleWeight,
                                 double &outPrice[],
                                 double &outCandle[],
                                 double &outZigZag[],
                                 double &outVolume[],
                                 double &outCombined[],
                                 double &outConfidence[],
                                 int &outFlags[]);
#import

bool RWIsBullish(const int flag)      { return (flag & RW_FLAG_BULLISH) != 0; }
bool RWIsBearish(const int flag)      { return (flag & RW_FLAG_BEARISH) != 0; }
bool RWIsLowConfidence(const int flag){ return (flag & RW_FLAG_LOW_CONFIDENCE) != 0; }
bool RWIsWarmup(const int flag)       { return (flag & RW_FLAG_WARMUP) != 0; }

input int    InWindowSize      = 192;
input int    InModeFlags       = RW_MODE_HIGHPASS | RW_MODE_EMPHASIZE_PIVOT | RW_MODE_USE_HANN;
input double InPriceWeight     = 0.7;
input double InVolumeWeight    = 0.2;
input double InPivotWeight     = 0.1;
input double InCandleWeight    = 0.3;
input int    InZigZagDepth     = 12;
input int    InZigZagDeviation = 5;
input int    InZigZagBackstep  = 3;

int g_zzHandle = INVALID_HANDLE;
double g_waveBuffer[];
double g_confidenceBuffer[];
double g_bullBuffer[];
double g_bearBuffer[];
double g_volumeSeries[];
double g_pivotSeries[];
double g_priceWave[];
double g_candleWave[];
double g_zzWave[];
double g_volumeWave[];
double g_combinedWave[];
double g_confScratch[];
int    g_flagScratch[];

double g_zzHigh[];
double g_zzLow[];

int OnInit()
{
   IndicatorSetInteger(INDICATOR_DIGITS, 4);
   SetIndexBuffer(0, g_waveBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, g_confidenceBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, g_bullBuffer, INDICATOR_DATA);
   SetIndexBuffer(3, g_bearBuffer, INDICATOR_DATA);
   // Buffers de cálculo (expostos a CopyBuffer via iCustom, mas não plotados)
   SetIndexBuffer(4, g_priceWave,   INDICATOR_CALCULATIONS);
   SetIndexBuffer(5, g_candleWave,  INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, g_zzWave,      INDICATOR_CALCULATIONS);
   SetIndexBuffer(7, g_volumeWave,  INDICATOR_CALCULATIONS);
   ArraySetAsSeries(g_waveBuffer, true);
   ArraySetAsSeries(g_confidenceBuffer, true);
   ArraySetAsSeries(g_bullBuffer, true);
   ArraySetAsSeries(g_bearBuffer, true);
   ArraySetAsSeries(g_priceWave, true);
   ArraySetAsSeries(g_candleWave, true);
   ArraySetAsSeries(g_zzWave, true);
   ArraySetAsSeries(g_volumeWave, true);
   PlotIndexSetString(0, PLOT_LABEL, "ReversalWave");
   PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, clrDodgerBlue);
   PlotIndexSetString(1, PLOT_LABEL, "Confidence");
   PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_NONE); // desenharemos apenas um traço por vez
   PlotIndexSetInteger(1, PLOT_LINE_COLOR, clrLimeGreen);
   PlotIndexSetString(2, PLOT_LABEL, "Bullish");
   PlotIndexSetInteger(2, PLOT_DRAW_TYPE, DRAW_NONE);
   PlotIndexSetInteger(2, PLOT_LINE_COLOR, clrSpringGreen);
   PlotIndexSetInteger(2, PLOT_ARROW, 233);
   PlotIndexSetInteger(3, PLOT_DRAW_TYPE, DRAW_NONE);
   PlotIndexSetString(3, PLOT_LABEL, "Bearish");
   PlotIndexSetInteger(3, PLOT_LINE_COLOR, clrTomato);
   PlotIndexSetInteger(3, PLOT_ARROW, 234);

   g_zzHandle = iCustom(_Symbol, _Period, "Examples\\ZigZag",
                        InZigZagDepth, InZigZagDeviation, InZigZagBackstep);
   if(g_zzHandle == INVALID_HANDLE)
   {
      PrintFormat("[ReversalWave] Failed to create ZigZag handle: %d", GetLastError());
      return INIT_FAILED;
   }

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_zzHandle != INVALID_HANDLE)
   {
      IndicatorRelease(g_zzHandle);
      g_zzHandle = INVALID_HANDLE;
   }
}

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
   if(rates_total < InWindowSize)
      return prev_calculated;

   const int total = rates_total;
   ArraySetAsSeries(g_volumeSeries, true);
   ArraySetAsSeries(g_pivotSeries, true);
   ArraySetAsSeries(g_priceWave, true);
   ArraySetAsSeries(g_candleWave, true);
   ArraySetAsSeries(g_zzWave, true);
   ArraySetAsSeries(g_volumeWave, true);
   ArraySetAsSeries(g_combinedWave, true);
   ArraySetAsSeries(g_confScratch, true);
   ArraySetAsSeries(g_flagScratch, true);

   ArrayResize(g_volumeSeries, total);
   ArrayResize(g_pivotSeries, total);
   ArrayResize(g_priceWave, total);
   ArrayResize(g_candleWave, total);
   ArrayResize(g_zzWave, total);
   ArrayResize(g_volumeWave, total);
   ArrayResize(g_combinedWave, total);
   ArrayResize(g_confScratch, total);
   ArrayResize(g_flagScratch, total);

   for(int i = 0; i < total; ++i)
   {
      g_volumeSeries[i] = (double)volume[i];
      g_pivotSeries[i] = 0.0;
      g_priceWave[i] = 0.0;
      g_candleWave[i] = 0.0;
      g_zzWave[i] = 0.0;
      g_volumeWave[i] = 0.0;
      g_combinedWave[i] = 0.0;
      g_confScratch[i] = 0.0;
      g_flagScratch[i] = 0;
      g_bullBuffer[i] = EMPTY_VALUE;
      g_bearBuffer[i] = EMPTY_VALUE;
   }

   ArraySetAsSeries(g_zzHigh, true);
   ArraySetAsSeries(g_zzLow, true);
   ArrayResize(g_zzHigh, total);
   ArrayResize(g_zzLow, total);
   const int copiedHigh = CopyBuffer(g_zzHandle, 0, 0, total, g_zzHigh);
   const int copiedLow = CopyBuffer(g_zzHandle, 1, 0, total, g_zzLow);
   if(copiedHigh == total && copiedLow == total)
   {
      for(int i = 0; i < total; ++i)
      {
         if(g_zzHigh[i] != 0.0)
            g_pivotSeries[i] = 1.0;
         else if(g_zzLow[i] != 0.0)
            g_pivotSeries[i] = -1.0;
      }
   }

   // Chama V2 diretamente; se falhar, V1.
   bool ok = true;
   int rc = gpu_reversal_wave_process_v2(open, close, high, low, g_volumeSeries, g_pivotSeries,
                                         total, InWindowSize, InModeFlags,
                                         InPriceWeight, InVolumeWeight, InPivotWeight, InCandleWeight,
                                         g_priceWave, g_candleWave, g_zzWave, g_volumeWave,
                                         g_combinedWave, g_confScratch, g_flagScratch);
   if(rc != RW_RESULT_OK)
   {
      rc = gpu_reversal_wave_process(close, g_volumeSeries, g_pivotSeries, total,
                                     InWindowSize, InModeFlags,
                                     InPriceWeight, InVolumeWeight, InPivotWeight,
                                     g_combinedWave, g_confScratch, g_flagScratch);
      ok = (rc == RW_RESULT_OK);
   }
   if(!ok)
      return prev_calculated;

   for(int i = 0; i < total; ++i)
   {
      g_waveBuffer[i] = g_combinedWave[i];
      g_confidenceBuffer[i] = g_confScratch[i];
      g_bullBuffer[i] = RWIsBullish(g_flagScratch[i]) ? g_combinedWave[i] : EMPTY_VALUE;
      g_bearBuffer[i] = RWIsBearish(g_flagScratch[i]) ? g_combinedWave[i] : EMPTY_VALUE;
   }

   // Debug: tail of DLL log into the MT5 Journal
   static datetime lastPrint = 0;
   if(TimeCurrent() != lastPrint) // avoid spamming every tick
   {
      PrintTailLog(5);
      lastPrint = TimeCurrent();
   }
   return total;
}

void PrintTailLog(const int lines)
{
   int handle = FileOpen("reversal_wave_debug.log", FILE_READ|FILE_TXT|FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   const int64 size = FileSize(handle);
   const int64 back = 4096;
   if(size > back)
      FileSeek(handle, size - back, SEEK_SET);
   string accum = "";
   while(!FileIsEnding(handle))
   {
      string s = FileReadString(handle);
      if(GetLastError() != 0) break;
      accum += s + "\n";
   }
   FileClose(handle);
   string parts[];
   int cnt = StringSplit(accum, "\n", parts);
   int start = MathMax(0, cnt - lines);
   for(int i = start; i < cnt; ++i)
   {
      if(parts[i] != "")
         PrintFormat("[RW DEBUG] %s", parts[i]);
   }
}
