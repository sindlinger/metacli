//+------------------------------------------------------------------+
//|                                           Indicator_Sample.mq5   |
//+------------------------------------------------------------------+
#property strict
#property indicator_chart_window
#property indicator_plots   1
#property indicator_buffers 1
#property indicator_label1  "EMA14"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrLime
#property indicator_width1  2

input int InpPeriod=14;

double Buf[];

int OnInit()
{
  if(InpPeriod<2){ InpPeriod=2; }
  SetIndexBuffer(0, Buf, INDICATOR_DATA);
  PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpPeriod);
  IndicatorShortName(StringFormat("EMA(%d)", InpPeriod));
  return(INIT_SUCCEEDED);
}

int OnCalculate(const int rates_total,const int prev_calculated,
                const datetime& time[],const double& open[],const double& high[],
                const double& low[],const double& close[],const long& tick_volume[],
                const long& volume[],const int& spread[])
{
  if(rates_total<InpPeriod) return(0);
  int start = (prev_calculated==0) ? 0 : prev_calculated-1;
  double k = 2.0/(InpPeriod+1.0);

  if(prev_calculated==0){
    int seed=rates_total-InpPeriod;
    if(seed<0) seed=0;
    double sma=0; int n=0;
    for(int j=seed+InpPeriod-1; j>=seed; j--){ sma+=close[j]; n++; }
    Buf[seed+InpPeriod-1]=sma/n;
    start=seed+InpPeriod;
    if(start>rates_total-1) start=rates_total-1;
  }

  for(int i=start;i<rates_total;i++){
    if(i+1<rates_total && Buf[i+1]!=0.0) Buf[i]=close[i]*k + Buf[i+1]*(1.0-k);
    else if(prev_calculated>0) Buf[i]=close[i];
  }
  return(rates_total);
}
