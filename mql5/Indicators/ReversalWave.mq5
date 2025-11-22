#property copyright "Reversal Wave GPU Pipeline"
#property link      "https://github.com/sindlinger/metacli"
#property version   "1.00"
#property strict
#property description "Wrapper indicator that marshals data to the GPU reversal wave DLL."
#property indicator_separate_window
#property indicator_plots 4

#include <ReversalWaveBridge.mqh>

input int    InWindowSize      = 192;
input int    InModeFlags       = RW_MODE_HIGHPASS | RW_MODE_EMPHASIZE_PIVOT | RW_MODE_USE_HANN;
input double InPriceWeight     = 0.7;
input double InVolumeWeight    = 0.2;
input double InPivotWeight     = 0.1;
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
double g_waveScratch[];
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
   ArraySetAsSeries(g_waveBuffer, true);
   ArraySetAsSeries(g_confidenceBuffer, true);
   ArraySetAsSeries(g_bullBuffer, true);
   ArraySetAsSeries(g_bearBuffer, true);
   PlotIndexSetString(0, PLOT_LABEL, "ReversalWave");
   PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, clrDodgerBlue);
   PlotIndexSetString(1, PLOT_LABEL, "Confidence");
   PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(1, PLOT_LINE_COLOR, clrLimeGreen);
   PlotIndexSetString(2, PLOT_LABEL, "Bullish");
   PlotIndexSetInteger(2, PLOT_DRAW_TYPE, DRAW_ARROW);
   PlotIndexSetInteger(2, PLOT_LINE_COLOR, clrSpringGreen);
   PlotIndexSetInteger(2, PLOT_ARROW, 233);
   PlotIndexSetInteger(3, PLOT_DRAW_TYPE, DRAW_ARROW);
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
   ArraySetAsSeries(g_waveScratch, true);
   ArraySetAsSeries(g_confScratch, true);
   ArraySetAsSeries(g_flagScratch, true);

   ArrayResize(g_volumeSeries, total);
   ArrayResize(g_pivotSeries, total);
   ArrayResize(g_waveScratch, total);
   ArrayResize(g_confScratch, total);
   ArrayResize(g_flagScratch, total);

   for(int i = 0; i < total; ++i)
   {
      g_volumeSeries[i] = (double)volume[i];
      g_pivotSeries[i] = 0.0;
      g_waveScratch[i] = 0.0;
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
         if(!DoubleIsZero(g_zzHigh[i]))
            g_pivotSeries[i] = 1.0;
         else if(!DoubleIsZero(g_zzLow[i]))
            g_pivotSeries[i] = -1.0;
      }
   }

   if(!RWComputeWave(close, g_volumeSeries, g_pivotSeries, total,
                     InWindowSize, InModeFlags,
                     InPriceWeight, InVolumeWeight, InPivotWeight,
                     g_waveScratch, g_confScratch, g_flagScratch))
   {
      return prev_calculated;
   }

   for(int i = 0; i < total; ++i)
   {
      g_waveBuffer[i] = g_waveScratch[i];
      g_confidenceBuffer[i] = g_confScratch[i];
      if(RWIsBullish(g_flagScratch[i]))
         g_bullBuffer[i] = g_waveScratch[i];
      else
         g_bullBuffer[i] = EMPTY_VALUE;

      if(RWIsBearish(g_flagScratch[i]))
         g_bearBuffer[i] = g_waveScratch[i];
      else
         g_bearBuffer[i] = EMPTY_VALUE;
   }

   return total;
}
