// ReversalWaveBridge.mqh
// Thin wrapper around the GPU reversal wave DLL. Keeps the MT5 indicator simple.
#pragma once

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
#import

inline string RWResultToString(const int code)
{
   switch(code)
   {
      case RW_RESULT_OK: return "RW_RESULT_OK";
      case RW_RESULT_INVALID_ARGUMENT: return "RW_RESULT_INVALID_ARGUMENT";
      case RW_RESULT_NOT_ENOUGH_DATA: return "RW_RESULT_NOT_ENOUGH_DATA";
      default: return StringFormat("RW_RESULT_%d", code);
   }
}

inline bool RWIsBullish(const int flag)      { return (flag & RW_FLAG_BULLISH) != 0; }
inline bool RWIsBearish(const int flag)      { return (flag & RW_FLAG_BEARISH) != 0; }
inline bool RWIsLowConfidence(const int flag){ return (flag & RW_FLAG_LOW_CONFIDENCE) != 0; }
inline bool RWIsWarmup(const int flag)       { return (flag & RW_FLAG_WARMUP) != 0; }

inline bool RWComputeWave(const double &price[],
                          const double &volume[],
                          const double &pivots[],
                          int length,
                          int window,
                          int modeFlags,
                          double priceWeight,
                          double volumeWeight,
                          double pivotWeight,
                          double &wave[],
                          double &confidence[],
                          int &flags[])
{
   const int rc = gpu_reversal_wave_process(price, volume, pivots, length, window, modeFlags,
                                            priceWeight, volumeWeight, pivotWeight,
                                            wave, confidence, flags);
   if(rc != RW_RESULT_OK)
   {
      PrintFormat("[ReversalWaveBridge] gpu_reversal_wave_process failed: %s (%d)",
                  RWResultToString(rc), rc);
      return false;
   }
   return true;
}
