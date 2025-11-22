//+------------------------------------------------------------------+
//| GPU_WaveViz_Solo_Cycles.mq5                                      |
//| Wrapper: lê os ciclos do GPU_WaveViz_Solo via iCustom            |
//| e exibe até 4 buffers de ciclo.                                  |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "1.001"
#property strict

#property indicator_separate_window
#property indicator_buffers 4
#property indicator_plots   4

#property indicator_label1  "Cycle1"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue

#property indicator_label2  "Cycle2"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDeepSkyBlue

#property indicator_label3  "Cycle3"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrAqua

#property indicator_label4  "Cycle4"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrSpringGreen

double g_bufCycle1_2[];
double g_bufCycle2_2[];
double g_bufCycle3_2[];
double g_bufCycle4_2[];

int    g_handleSolo2 = INVALID_HANDLE;

int OnInit()
  {
   SetIndexBuffer(0, g_bufCycle1_2, INDICATOR_DATA);
   SetIndexBuffer(1, g_bufCycle2_2, INDICATOR_DATA);
   SetIndexBuffer(2, g_bufCycle3_2, INDICATOR_DATA);
   SetIndexBuffer(3, g_bufCycle4_2, INDICATOR_DATA);

   ArraySetAsSeries(g_bufCycle1_2, true);
   ArraySetAsSeries(g_bufCycle2_2, true);
   ArraySetAsSeries(g_bufCycle3_2, true);
   ArraySetAsSeries(g_bufCycle4_2, true);

   IndicatorSetString(INDICATOR_SHORTNAME, "GPU WaveViz Solo - CyclesView");

   g_handleSolo2 = iCustom(_Symbol, _Period, "Hub_WavePhaseSD/GPU_WaveViz_Solo");
   if(g_handleSolo2 == INVALID_HANDLE)
     {
      Print("[WaveViz Solo CyclesView] iCustom(GPU_WaveViz_Solo) falhou, err=", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(g_handleSolo2 != INVALID_HANDLE)
     {
      IndicatorRelease(g_handleSolo2);
      g_handleSolo2 = INVALID_HANDLE;
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
   if(g_handleSolo2 == INVALID_HANDLE || rates_total <= 0)
      return(prev_calculated);

   const int copied1 = CopyBuffer(g_handleSolo2, 2, 0, rates_total, g_bufCycle1_2);
   const int copied2 = CopyBuffer(g_handleSolo2, 3, 0, rates_total, g_bufCycle2_2);
   const int copied3 = CopyBuffer(g_handleSolo2, 4, 0, rates_total, g_bufCycle3_2);
   const int copied4 = CopyBuffer(g_handleSolo2, 5, 0, rates_total, g_bufCycle4_2);

   if(copied1 <= 0 && copied2 <= 0 && copied3 <= 0 && copied4 <= 0)
      return(prev_calculated);

   return(rates_total);
  }
