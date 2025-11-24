// ReversalWaveBridge.mqh
// Somente as imports das funções do DLL.
#pragma once

#define RW_RESULT_OK                 0
#define RW_RESULT_INVALID_ARGUMENT  -1
#define RW_RESULT_NOT_ENOUGH_DATA   -2

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

inline string RWResultToString(const int rc)
{
   switch(rc)
   {
      case RW_RESULT_OK:                return "OK";
      case RW_RESULT_INVALID_ARGUMENT:  return "INVALID_ARGUMENT";
      case RW_RESULT_NOT_ENOUGH_DATA:   return "NOT_ENOUGH_DATA";
      default:                          return "UNKNOWN";
   }
}
