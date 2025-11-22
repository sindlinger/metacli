//+------------------------------------------------------------------+
//| GPU_WaveViz_Solo_Kalman.mq5                                      |
//| Wrapper: lê buffers do GPU_WaveViz_Solo via iCustom              |
//| e exibe Wave (buf0) + uma cópia (Kalman view).                   |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "1.001"
#property strict

#property indicator_separate_window
#property indicator_buffers 2
#property indicator_plots   2

#property indicator_label1  "Wave"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrGold
#property indicator_width1  2

#property indicator_label2  "KalmanView"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDeepSkyBlue
#property indicator_width2  2

double g_bufWave[];
double g_bufKalman[];

int    g_handleSolo = INVALID_HANDLE;

int OnInit()
  {
   SetIndexBuffer(0, g_bufWave,   INDICATOR_DATA);
   SetIndexBuffer(1, g_bufKalman, INDICATOR_DATA);

   ArraySetAsSeries(g_bufWave,   true);
   ArraySetAsSeries(g_bufKalman, true);

   IndicatorSetString(INDICATOR_SHORTNAME, "GPU WaveViz Solo - KalmanView");

   g_handleSolo = iCustom(_Symbol, _Period, "Hub_WavePhaseSD/GPU_WaveViz_Solo");
   if(g_handleSolo == INVALID_HANDLE)
     {
      Print("[WaveViz Solo KalmanView] iCustom(GPU_WaveViz_Solo) falhou, err=", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(g_handleSolo != INVALID_HANDLE)
     {
      IndicatorRelease(g_handleSolo);
      g_handleSolo = INVALID_HANDLE;
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
   if(g_handleSolo == INVALID_HANDLE || rates_total <= 0)
      return(prev_calculated);

   const int copiedWave   = CopyBuffer(g_handleSolo, 0, 0, rates_total, g_bufWave);
   const int copiedWave2  = CopyBuffer(g_handleSolo, 0, 0, rates_total, g_bufKalman);

   if(copiedWave <= 0 || copiedWave2 <= 0)
      return(prev_calculated);

   return(rates_total);
  }
