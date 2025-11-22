#property script_show_inputs
#property strict

#include <ReversalWaveBridge.mqh>

input int    InSampleCount = 256;
input double InOscillation = 48.0;
input double InNoise       = 0.25;

void OnStart()
{
   if(InSampleCount < 64)
   {
      Print("[ReversalWaveSelfTest] Increase InSampleCount to >= 64");
      return;
   }
   double wave[];
   double confidence[];
   int flags[];
   ArraySetAsSeries(wave, true);
   ArraySetAsSeries(confidence, true);
   ArraySetAsSeries(flags, true);
   ArrayResize(wave, InSampleCount);
   ArrayResize(confidence, InSampleCount);
   ArrayResize(flags, InSampleCount);

   const int rc = gpu_reversal_wave_synthetic_test(InSampleCount, InOscillation, InNoise,
                                                   wave, confidence, flags);
   if(rc != RW_RESULT_OK)
   {
      PrintFormat("[ReversalWaveSelfTest] gpu_reversal_wave_synthetic_test failed: %s (%d)",
                  RWResultToString(rc), rc);
      return;
   }

   double avgConfidence = 0.0;
   int bullish = 0;
   int bearish = 0;
   int lowConf = 0;
   int warmup = 0;
   for(int i = 0; i < InSampleCount; ++i)
   {
      avgConfidence += confidence[i];
      if(RWIsBullish(flags[i]))
         ++bullish;
      if(RWIsBearish(flags[i]))
         ++bearish;
      if(RWIsLowConfidence(flags[i]))
         ++lowConf;
      if(RWIsWarmup(flags[i]))
         ++warmup;
   }
   avgConfidence /= (double)InSampleCount;
   PrintFormat("[ReversalWaveSelfTest] length=%d, avg_conf=%.3f, bull=%d, bear=%d, low_conf=%d, warmup=%d",
               InSampleCount, avgConfidence, bullish, bearish, lowConf, warmup);
}
