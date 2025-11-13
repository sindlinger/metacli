//+------------------------------------------------------------------+
//|                                                 EA_Sample.mq5    |
//+------------------------------------------------------------------+
#property strict
input int InpMAPeriod=20;
int ma_handle=INVALID_HANDLE;
double buf[3];

int OnInit(){
  if(InpMAPeriod<2) InpMAPeriod=2;
  ma_handle=iMA(_Symbol,_Period,InpMAPeriod,0,MODE_EMA,PRICE_CLOSE);
  if(ma_handle==INVALID_HANDLE){ Print("iMA failed: ",GetLastError()); return(INIT_FAILED); }
  return(INIT_SUCCEEDED);
}
void OnDeinit(const int reason){
  if(ma_handle!=INVALID_HANDLE){ IndicatorRelease(ma_handle); ma_handle=INVALID_HANDLE; }
}
void OnTick(){
  static datetime last_bar=0;
  datetime t=iTime(_Symbol,_Period,0);
  if(t==last_bar) return; last_bar=t;
  if(BarsCalculated(ma_handle)<=0) return;
  if(CopyBuffer(ma_handle,0,0,2,buf)<=0){ Print("CopyBuffer err:",GetLastError()); return; }
  double cur=buf[1], prev=buf[1];
  string bias = (cur>prev) ? "alta" : (cur<prev ? "baixa":"neutra");
  Print(StringFormat("MA(%d) %.5f dir=%s", InpMAPeriod, cur, bias));
}
