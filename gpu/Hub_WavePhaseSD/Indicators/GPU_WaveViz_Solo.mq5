//+------------------------------------------------------------------+
//| GPU_WaveViz_Solo.mq5                                            |
//| Visualiza a saída da GPU sem depender do hub.                   |
//| Calcula STFT + ciclos internamente (1 frame por submissão).     |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "1.100"
#property strict

#property indicator_separate_window
#property indicator_buffers 26
#property indicator_plots   26

#property indicator_label1  "Wave"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrGold
#property indicator_width1  2

#property indicator_label2  "Noise"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrSilver
#property indicator_width2  1

#property indicator_label3  "Cycle1"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrDodgerBlue

#property indicator_label4  "Cycle2"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrDeepSkyBlue

#property indicator_label5  "Cycle3"
#property indicator_type5   DRAW_LINE
#property indicator_color5  clrAqua

#property indicator_label6  "Cycle4"
#property indicator_type6   DRAW_LINE
#property indicator_color6  clrSpringGreen

#property indicator_label7  "Cycle5"
#property indicator_type7   DRAW_LINE
#property indicator_color7  clrGreen

#property indicator_label8  "Cycle6"
#property indicator_type8   DRAW_LINE
#property indicator_color8  clrYellowGreen

#property indicator_label9  "Cycle7"
#property indicator_type9   DRAW_LINE
#property indicator_color9  clrOrange

#property indicator_label10 "Cycle8"
#property indicator_type10  DRAW_LINE
#property indicator_color10 clrTomato

#property indicator_label11 "Cycle9"
#property indicator_type11  DRAW_LINE
#property indicator_color11 clrCrimson

#property indicator_label12 "Cycle10"
#property indicator_type12  DRAW_LINE
#property indicator_color12 clrViolet

#property indicator_label13 "Cycle11"
#property indicator_type13  DRAW_LINE
#property indicator_color13 clrMagenta

#property indicator_label14 "Cycle12"
#property indicator_type14  DRAW_LINE
#property indicator_color14 clrSlateBlue

#property indicator_label15 "Cycle13"
#property indicator_type15  DRAW_LINE
#property indicator_color15 clrOrangeRed

#property indicator_label16 "Cycle14"
#property indicator_type16  DRAW_LINE
#property indicator_color16 clrLime

#property indicator_label17 "Cycle15"
#property indicator_type17  DRAW_LINE
#property indicator_color17 clrSkyBlue

#property indicator_label18 "Cycle16"
#property indicator_type18  DRAW_LINE
#property indicator_color18 clrOrange

#property indicator_label19 "Cycle17"
#property indicator_type19  DRAW_LINE
#property indicator_color19 clrGold

#property indicator_label20 "Cycle18"
#property indicator_type20  DRAW_LINE
#property indicator_color20 clrDarkTurquoise

#property indicator_label21 "Cycle19"
#property indicator_type21  DRAW_LINE
#property indicator_color21 clrPaleGreen

#property indicator_label22 "Cycle20"
#property indicator_type22  DRAW_LINE
#property indicator_color22 clrMediumSlateBlue

#property indicator_label23 "Cycle21"
#property indicator_type23  DRAW_LINE
#property indicator_color23 clrDeepPink

#property indicator_label24 "Cycle22"
#property indicator_type24  DRAW_LINE
#property indicator_color24 clrCornflowerBlue

#property indicator_label25 "Cycle23"
#property indicator_type25  DRAW_LINE
#property indicator_color25 clrKhaki

#property indicator_label26 "Cycle24"
#property indicator_type26  DRAW_LINE
#property indicator_color26 clrMediumPurple

#include <GPU/GPU_Engine.mqh>

enum SoloFeedMode
  {
   Feed_Close = 0
  };

input bool        InpVerboseLog      = true;
input int         InpGPUDevice       = 0;
input int         InpFFTWindow       = 4096;
input int         InpHop             = 1024;
input bool        InpUseManualCycles = false;
input int         InpCycleCount      = 24;
input double      InpCycleMinPeriod  = 18.0;
input double      InpCycleMaxPeriod  = 1440.0;
input double      InpCycleWidth      = 0.25;
input int         InpMaxCandidates   = 24;
input double      InpGaussSigmaPeriod       = 48.0;
input double      InpMaskThreshold   = 0.05;
input double      InpMaskSoftness    = 0.20;
input double      InpMaskMinPeriod   = 18.0;
input double      InpMaskMaxPeriod   = 512.0;

enum KalmanPresetOption
  {
   KalmanSmooth   = 0,
   KalmanBalanced = 1,
   KalmanReactive = 2,
   KalmanManual   = 3
  };

input KalmanPresetOption InpKalmanPreset         = KalmanBalanced;
input double             InpKalmanProcessNoise   = 1.0e-4;
input double             InpKalmanMeasurementNoise = 2.5e-3;
input double             InpKalmanInitVariance   = 0.5;
input double             InpKalmanPlvThreshold   = 0.35;
input int                InpKalmanMaxIterations  = 48;
input double             InpKalmanConvergenceEps = 1.0e-4;

input SoloFeedMode InpFeedMode       = Feed_Close;
input bool        InpShowNoise       = true;
input bool        InpShowCycles      = true;

const uint JOB_FLAG_STFT   = 1;
const uint JOB_FLAG_CYCLES = 2;

double g_bufWave[];
double g_bufNoise[];
double g_bufCycle1[];
double g_bufCycle2[];
double g_bufCycle3[];
double g_bufCycle4[];
double g_bufCycle5[];
double g_bufCycle6[];
double g_bufCycle7[];
double g_bufCycle8[];
double g_bufCycle9[];
double g_bufCycle10[];
double g_bufCycle11[];
double g_bufCycle12[];
double g_bufCycle13[];
double g_bufCycle14[];
double g_bufCycle15[];
double g_bufCycle16[];
double g_bufCycle17[];
double g_bufCycle18[];
double g_bufCycle19[];
double g_bufCycle20[];
double g_bufCycle21[];
double g_bufCycle22[];
double g_bufCycle23[];
double g_bufCycle24[];

CGpuEngineClient g_engine;
bool              g_prevLogging      = true;

struct PendingJob
  {
   bool  active;
   ulong handle;
   ulong tag;
   int   submitted_bars;
  };

PendingJob g_job = { false, 0, 0, -1 };

double g_frames[];
double g_cyclePeriods[];
string g_statusText = "";
double g_waveOut[];
double g_previewOut[];
double g_cyclesOut[];
double g_noiseOut[];
double g_phaseOut[];
double g_phaseUnwrappedOut[];
double g_amplitudeOut[];
double g_periodOut[];
double g_frequencyOut[];
double g_etaOut[];
double g_countdownOut[];
double g_reconOut[];
double g_kalmanOut[];
double g_confidenceOut[];
double g_ampDeltaOut[];
double g_turnOut[];
double g_directionOut[];
double g_powerOut[];
double g_velocityOut[];

double g_phaseAllOut[];
double g_phaseUnwrappedAllOut[];
double g_amplitudeAllOut[];
double g_periodAllOut[];
double g_frequencyAllOut[];
double g_etaAllOut[];
double g_countdownAllOut[];
double g_directionAllOut[];
double g_reconAllOut[];
double g_kalmanAllOut[];
double g_turnAllOut[];
double g_confidenceAllOut[];
double g_ampDeltaAllOut[];
double g_powerAllOut[];
double g_velocityAllOut[];

double g_plvCyclesOut[];
double g_snrCyclesOut[];

GpuEngineResultInfo g_lastInfo;

int  g_lastFrameLength = 0;
bool g_hasWaveResult   = false;
double g_initProgress  = 0.0;

void ClearCycleBuffers()
  {
   ArrayInitialize(g_bufCycle1,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle2,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle3,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle4,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle5,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle6,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle7,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle8,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle9,  EMPTY_VALUE);
   ArrayInitialize(g_bufCycle10, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle11, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle12, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle13, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle14, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle15, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle16, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle17, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle18, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle19, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle20, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle21, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle22, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle23, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle24, EMPTY_VALUE);
  }

void FillWaveNoiseFromPriceFallback(const int rates_total,
                                    const double &close[])
  {
   if(rates_total <= 0)
      return;

   const int bars = MathMin(rates_total, InpFFTWindow);
   for(int i=0; i<bars; ++i)
     {
      // close[] já está em série (índice 0 = barra mais recente).
      g_bufWave[i]  = close[i];
      g_bufNoise[i] = 0.0;
     }
  }

void SetCycleValue(const int index,
                   const int bar_index,
                   const double value)
  {
   switch(index)
     {
      case 0: g_bufCycle1[bar_index]  = value; break;
      case 1: g_bufCycle2[bar_index]  = value; break;
      case 2: g_bufCycle3[bar_index]  = value; break;
      case 3: g_bufCycle4[bar_index]  = value; break;
      case 4: g_bufCycle5[bar_index]  = value; break;
      case 5: g_bufCycle6[bar_index]  = value; break;
      case 6: g_bufCycle7[bar_index]  = value; break;
      case 7: g_bufCycle8[bar_index]  = value; break;
      case 8: g_bufCycle9[bar_index]  = value; break;
      case 9: g_bufCycle10[bar_index] = value; break;
      case 10:g_bufCycle11[bar_index] = value; break;
      case 11:g_bufCycle12[bar_index] = value; break;
      case 12:g_bufCycle13[bar_index] = value; break;
      case 13:g_bufCycle14[bar_index] = value; break;
      case 14:g_bufCycle15[bar_index] = value; break;
      case 15:g_bufCycle16[bar_index] = value; break;
      case 16:g_bufCycle17[bar_index] = value; break;
      case 17:g_bufCycle18[bar_index] = value; break;
      case 18:g_bufCycle19[bar_index] = value; break;
      case 19:g_bufCycle20[bar_index] = value; break;
      case 20:g_bufCycle21[bar_index] = value; break;
      case 21:g_bufCycle22[bar_index] = value; break;
      case 22:g_bufCycle23[bar_index] = value; break;
      case 23:g_bufCycle24[bar_index] = value; break;
     }
  }

void ReverseArray(double &data[])
  {
   const int size = ArraySize(data);
   for(int i=0, j=size-1; i<j; ++i, --j)
     {
      const double tmp = data[i];
      data[i] = data[j];
      data[j] = tmp;
     }
  }

void BuildCyclePeriods()
  {
   static const double defaults[24] = {
      18.0, 24.0, 30.0, 36.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0, 180.0, 240.0,
      300.0, 360.0, 420.0, 480.0, 540.0, 600.0, 720.0, 840.0, 960.0, 1080.0, 1260.0, 1440.0
   };
   const int max_defaults = ArraySize(defaults);
   ArrayResize(g_cyclePeriods, max_defaults);
   for(int i=0; i<max_defaults; ++i)
      g_cyclePeriods[i] = defaults[i];

   if(!InpUseManualCycles)
     {
      int limit = (int)MathMax(MathMin((double)InpMaxCandidates, (double)max_defaults), 0.0);
      if(limit > 0 && limit < max_defaults)
         ArrayResize(g_cyclePeriods, limit);
      return;
     }

   const int count = (int)MathMax(MathMin((double)InpCycleCount, (double)max_defaults), 0.0);
   if(count <= 0)
      return;

   ArrayResize(g_cyclePeriods, count);
   const double minP = MathMax(InpCycleMinPeriod, 1.0);
   const double maxP = MathMax(minP, InpCycleMaxPeriod);
   if(count == 1)
     {
      g_cyclePeriods[0] = minP;
      return;
     }
   const double ratio = MathPow(maxP / minP, 1.0 / (count - 1));
   double value = minP;
   for(int i=0; i<count; ++i)
     {
      g_cyclePeriods[i] = value;
      value *= ratio;
     }
  }


bool PrepareFrames()
  {
   ArrayResize(g_frames, InpFFTWindow);
   ArraySetAsSeries(g_frames, true);

   int copied = 0;
   switch(InpFeedMode)
     {
      case Feed_Close:
         copied = CopyClose(_Symbol, _Period, 0, InpFFTWindow, g_frames);
         break;
     }

   if(copied != InpFFTWindow)
      return(false);

   ArraySetAsSeries(g_frames, false);
   ReverseArray(g_frames);
   return(true);
  }

bool SubmitJob(const int rates_total)
  {
   if(!PrepareFrames())
      return(false);

   BuildCyclePeriods();

   ulong handle = 0;
   const ulong tag = ++g_job.tag;

   int cycle_count = (int)MathMin((double)ArraySize(g_cyclePeriods), 24.0);
   if(cycle_count < 0)
      cycle_count = 0;

   uint flags;
   bool submitted;
   if(cycle_count > 0)
     {
      flags = JOB_FLAG_STFT | JOB_FLAG_CYCLES;
      submitted = g_engine.SubmitJobEx(g_frames,
                                       1,
                                       tag,
                                       flags,
                                       g_gpuEmptyPreviewMask,
                                       g_cyclePeriods,
                                       cycle_count,
                                       InpCycleWidth,
                                       InpGaussSigmaPeriod,
                                       InpMaskThreshold,
                                       InpMaskSoftness,
                                       InpMaskMinPeriod,
                                       InpMaskMaxPeriod,
                                       1,
                                       (int)InpKalmanPreset,
                                       InpKalmanProcessNoise,
                                       InpKalmanMeasurementNoise,
                                       InpKalmanInitVariance,
                                       InpKalmanPlvThreshold,
                                       InpKalmanMaxIterations,
                                       InpKalmanConvergenceEps,
                                       handle);
     }
   else
     {
      flags = JOB_FLAG_STFT;
      submitted = g_engine.SubmitJobEx(g_frames,
                                       1,
                                       tag,
                                       flags,
                                       g_gpuEmptyPreviewMask,
                                       g_gpuEmptyCyclePeriods,
                                       0,
                                       InpCycleWidth,
                                       InpGaussSigmaPeriod,
                                       InpMaskThreshold,
                                       InpMaskSoftness,
                                       InpMaskMinPeriod,
                                       InpMaskMaxPeriod,
                                       1,
                                       (int)InpKalmanPreset,
                                       InpKalmanProcessNoise,
                                       InpKalmanMeasurementNoise,
                                       InpKalmanInitVariance,
                                       InpKalmanPlvThreshold,
                                       InpKalmanMaxIterations,
                                       InpKalmanConvergenceEps,
                                       handle);
     }

   if(!submitted)
      return(false);

   g_job.active          = true;
   g_job.handle          = handle;
   g_job.submitted_bars  = rates_total;
   return(true);
  }

void EnsureFetchBuffers(const int frame_total,
                        const int cycle_total,
                        const int cycle_count)
  {
   const int safe_frame_total = (frame_total > 0 ? frame_total : 1);
   const int safe_cycle_total = (cycle_total > 0 ? cycle_total : 1);
   const int safe_cycle_count = (cycle_count > 0 ? cycle_count : 1);

   ArrayResize(g_waveOut,    safe_frame_total);
   ArrayResize(g_previewOut, safe_frame_total);
   ArrayResize(g_noiseOut,   safe_frame_total);
   ArrayResize(g_cyclesOut,  safe_cycle_total);

   ArrayResize(g_phaseOut,             safe_frame_total);
   ArrayResize(g_phaseUnwrappedOut,    safe_frame_total);
   ArrayResize(g_amplitudeOut,         safe_frame_total);
   ArrayResize(g_periodOut,            safe_frame_total);
   ArrayResize(g_frequencyOut,         safe_frame_total);
   ArrayResize(g_etaOut,               safe_frame_total);
   ArrayResize(g_countdownOut,         safe_frame_total);
   ArrayResize(g_reconOut,             safe_frame_total);
   ArrayResize(g_kalmanOut,            safe_frame_total);
   ArrayResize(g_confidenceOut,        safe_frame_total);
   ArrayResize(g_ampDeltaOut,          safe_frame_total);
   ArrayResize(g_turnOut,              safe_frame_total);
   ArrayResize(g_directionOut,         safe_frame_total);
   ArrayResize(g_powerOut,             safe_frame_total);
   ArrayResize(g_velocityOut,          safe_frame_total);

   ArrayResize(g_phaseAllOut,          safe_cycle_total);
   ArrayResize(g_phaseUnwrappedAllOut, safe_cycle_total);
   ArrayResize(g_amplitudeAllOut,      safe_cycle_total);
   ArrayResize(g_periodAllOut,         safe_cycle_total);
   ArrayResize(g_frequencyAllOut,      safe_cycle_total);
   ArrayResize(g_etaAllOut,            safe_cycle_total);
   ArrayResize(g_countdownAllOut,      safe_cycle_total);
   ArrayResize(g_directionAllOut,      safe_cycle_total);
   ArrayResize(g_reconAllOut,          safe_cycle_total);
   ArrayResize(g_kalmanAllOut,         safe_cycle_total);
   ArrayResize(g_turnAllOut,           safe_cycle_total);
   ArrayResize(g_confidenceAllOut,     safe_cycle_total);
   ArrayResize(g_ampDeltaAllOut,       safe_cycle_total);
   ArrayResize(g_powerAllOut,          safe_cycle_total);
   ArrayResize(g_velocityAllOut,       safe_cycle_total);

   ArrayResize(g_plvCyclesOut, safe_cycle_count);
   ArrayResize(g_snrCyclesOut, safe_cycle_count);
  }

void CopyResultsToBuffers(const GpuEngineResultInfo &info)
  {
   const int frame_length = info.frame_length;
   const int frame_count  = info.frame_count;
   const int total_span   = frame_length * frame_count;
   const int total_cycle_count = info.cycle_count;

   if(frame_length <= 0 || frame_count <= 0)
      return;

   ArrayInitialize(g_bufWave,  EMPTY_VALUE);
   ArrayInitialize(g_bufNoise, EMPTY_VALUE);
   ClearCycleBuffers();

   if(ArraySize(g_waveOut)  < total_span ||
      ArraySize(g_noiseOut) < total_span)
      return;

   g_lastFrameLength = frame_length;

   const int samples_total = frame_length;
   const int frame_offset  = (frame_count - 1) * frame_length;

   // Seleção de ciclos (máx. 4), priorizando PLV quando disponível.
   const int max_visual_cycles = 4;
   const int requested_cycles  = (InpShowCycles ? max_visual_cycles : 0);
   const int target_cycle_count = (int)MathMin(MathMax(requested_cycles, 0), total_cycle_count);

   int  selected_count = 0;
   int  selected_indices[];
   bool has_plv = (total_cycle_count > 0 && ArraySize(g_plvCyclesOut) >= total_cycle_count);

   if(target_cycle_count > 0)
     {
      ArrayResize(selected_indices, target_cycle_count);

      if(has_plv)
        {
         bool used[];
         ArrayResize(used, total_cycle_count);
         ArrayInitialize(used, false);

         for(int rank=0; rank<target_cycle_count; ++rank)
           {
            double best = -DBL_MAX;
            int    best_idx = -1;
            for(int c=0; c<total_cycle_count; ++c)
              {
               if(used[c])
                  continue;
               double plv_val = g_plvCyclesOut[c];
               if(plv_val > best)
                 {
                  best     = plv_val;
                  best_idx = c;
                 }
              }
            if(best_idx < 0)
               break;
            selected_indices[selected_count++] = best_idx;
            used[best_idx] = true;
           }

         // Fallback: se a PLV não destacar ciclos suficientes, preenche o restante na ordem natural.
         if(selected_count < target_cycle_count)
           {
            bool used_fallback[];
            ArrayResize(used_fallback, total_cycle_count);
            ArrayInitialize(used_fallback, false);

            for(int i=0; i<selected_count; ++i)
              {
               const int idx = selected_indices[i];
               if(idx >= 0 && idx < total_cycle_count)
                  used_fallback[idx] = true;
              }

            for(int c=0; c<total_cycle_count && selected_count < target_cycle_count; ++c)
              {
               if(used_fallback[c])
                  continue;
               selected_indices[selected_count++] = c;
               used_fallback[c] = true;
              }
           }
        }
      else
        {
         for(int rank=0; rank<target_cycle_count; ++rank)
           {
            selected_indices[selected_count++] = rank;
           }
        }

      ArrayResize(selected_indices, selected_count);
     }

   // Copia a wave e o ruído do último frame.
   for(int i=0; i<samples_total; ++i)
     {
      const int src_index  = frame_offset + i;
      const int dest_index = i;
      g_bufWave[dest_index] = g_waveOut[src_index];
      if(InpShowNoise)
         g_bufNoise[dest_index] = g_noiseOut[src_index];

      if(InpShowCycles && selected_count > 0)
        {
         const int needed = total_span * total_cycle_count;
         bool has_recon_all  = (ArraySize(g_reconAllOut) >= needed);
         bool has_cycles_all = (ArraySize(g_cyclesOut)   >= needed);

         for(int rank=0; rank<selected_count && rank<max_visual_cycles; ++rank)
           {
            const int cyc_idx    = selected_indices[rank];
            const int cycle_base = cyc_idx * total_span;
            double value = EMPTY_VALUE;

            if(has_recon_all)
               value = g_reconAllOut[cycle_base + src_index];
            else if(has_cycles_all)
               value = g_cyclesOut[cycle_base + src_index];

            SetCycleValue(rank, dest_index, value);
           }
        }
     }

   // A partir daqui consideramos que há pelo menos um frame válido vindo da GPU.
   g_hasWaveResult = true;
   g_initProgress  = 1.0;

   if(InpVerboseLog)
     {
      const double wave_last   = (samples_total > 0 ? g_bufWave[0] : 0.0);
      const double cycle1_last = (samples_total > 0 ? g_bufCycle1[0] : 0.0);
      PrintFormat("[WaveViz Solo][DBG] frame_len=%d cycles_drawn=%d wave_first=%.6f cycle1_first=%.6f",
                  samples_total,
                  selected_count,
                  wave_last,
                  cycle1_last);
     }
  }

bool FetchCurrentResult()
  {
   int status = GPU_ENGINE_IN_PROGRESS;
   if(g_engine.PollStatus(g_job.handle, status) != GPU_ENGINE_OK)
      return(false);

   if(status != GPU_ENGINE_READY)
      return(false);

   const int frame_total = InpFFTWindow;
   const bool use_manual_cycles = InpUseManualCycles && ArraySize(g_cyclePeriods) > 0;
   const int max_cycles = (use_manual_cycles ? (int)MathMin((double)ArraySize(g_cyclePeriods), 24.0)
                                            : (int)MathMax(MathMin((double)InpMaxCandidates, 24.0), 0.0));
   const int safe_cycles = (max_cycles > 0 ? max_cycles : 0);
   EnsureFetchBuffers(frame_total, safe_cycles * frame_total, max_cycles);

   if(!g_engine.FetchResult(g_job.handle,
                            g_waveOut,
                            g_previewOut,
                            g_cyclesOut,
                            g_noiseOut,
                            g_phaseOut,
                            g_phaseUnwrappedOut,
                            g_amplitudeOut,
                            g_periodOut,
                            g_frequencyOut,
                            g_etaOut,
                            g_countdownOut,
                            g_reconOut,
                            g_kalmanOut,
                            g_confidenceOut,
                            g_ampDeltaOut,
                            g_turnOut,
                            g_directionOut,
                            g_powerOut,
                            g_velocityOut,
                            g_phaseAllOut,
                            g_phaseUnwrappedAllOut,
                            g_amplitudeAllOut,
                            g_periodAllOut,
                            g_frequencyAllOut,
                            g_etaAllOut,
                            g_countdownAllOut,
                            g_directionAllOut,
                            g_reconAllOut,
                            g_kalmanAllOut,
                            g_turnAllOut,
                            g_confidenceAllOut,
                            g_ampDeltaAllOut,
                            g_powerAllOut,
                            g_velocityAllOut,
                            g_plvCyclesOut,
                            g_snrCyclesOut,
                            g_lastInfo))
      return(false);

   CopyResultsToBuffers(g_lastInfo);
   g_statusText = StringFormat("CycleCount=%d DominantPLV=%.3f", g_lastInfo.cycle_count, g_lastInfo.dominant_plv);
   Comment(g_statusText);
   g_job.active = false;
   return(true);
  }


int OnInit()
  {
   SetIndexBuffer(0, g_bufWave,  INDICATOR_DATA);
   SetIndexBuffer(1, g_bufNoise, INDICATOR_DATA);
   SetIndexBuffer(2, g_bufCycle1,  INDICATOR_DATA);
   SetIndexBuffer(3, g_bufCycle2,  INDICATOR_DATA);
   SetIndexBuffer(4, g_bufCycle3,  INDICATOR_DATA);
   SetIndexBuffer(5, g_bufCycle4,  INDICATOR_DATA);
   SetIndexBuffer(6, g_bufCycle5,  INDICATOR_DATA);
   SetIndexBuffer(7, g_bufCycle6,  INDICATOR_DATA);
   SetIndexBuffer(8, g_bufCycle7,  INDICATOR_DATA);
   SetIndexBuffer(9, g_bufCycle8,  INDICATOR_DATA);
   SetIndexBuffer(10,g_bufCycle9,  INDICATOR_DATA);
   SetIndexBuffer(11,g_bufCycle10, INDICATOR_DATA);
   SetIndexBuffer(12,g_bufCycle11, INDICATOR_DATA);
   SetIndexBuffer(13,g_bufCycle12, INDICATOR_DATA);
   SetIndexBuffer(14,g_bufCycle13, INDICATOR_DATA);
   SetIndexBuffer(15,g_bufCycle14, INDICATOR_DATA);
   SetIndexBuffer(16,g_bufCycle15, INDICATOR_DATA);
   SetIndexBuffer(17,g_bufCycle16, INDICATOR_DATA);
   SetIndexBuffer(18,g_bufCycle17, INDICATOR_DATA);
   SetIndexBuffer(19,g_bufCycle18, INDICATOR_DATA);
   SetIndexBuffer(20,g_bufCycle19, INDICATOR_DATA);
   SetIndexBuffer(21,g_bufCycle20, INDICATOR_DATA);
   SetIndexBuffer(22,g_bufCycle21, INDICATOR_DATA);
   SetIndexBuffer(23,g_bufCycle22, INDICATOR_DATA);
   SetIndexBuffer(24,g_bufCycle23, INDICATOR_DATA);
   SetIndexBuffer(25,g_bufCycle24, INDICATOR_DATA);

   ArraySetAsSeries(g_bufWave,  true);
   ArraySetAsSeries(g_bufNoise, true);
   ArraySetAsSeries(g_bufCycle1,  true);
   ArraySetAsSeries(g_bufCycle2,  true);
   ArraySetAsSeries(g_bufCycle3,  true);
   ArraySetAsSeries(g_bufCycle4,  true);
   ArraySetAsSeries(g_bufCycle5,  true);
   ArraySetAsSeries(g_bufCycle6,  true);
   ArraySetAsSeries(g_bufCycle7,  true);
   ArraySetAsSeries(g_bufCycle8,  true);
   ArraySetAsSeries(g_bufCycle9,  true);
   ArraySetAsSeries(g_bufCycle10, true);
   ArraySetAsSeries(g_bufCycle11, true);
   ArraySetAsSeries(g_bufCycle12, true);
   ArraySetAsSeries(g_bufCycle13, true);
   ArraySetAsSeries(g_bufCycle14, true);
   ArraySetAsSeries(g_bufCycle15, true);
   ArraySetAsSeries(g_bufCycle16, true);
   ArraySetAsSeries(g_bufCycle17, true);
   ArraySetAsSeries(g_bufCycle18, true);
   ArraySetAsSeries(g_bufCycle19, true);
   ArraySetAsSeries(g_bufCycle20, true);
   ArraySetAsSeries(g_bufCycle21, true);
   ArraySetAsSeries(g_bufCycle22, true);
   ArraySetAsSeries(g_bufCycle23, true);
   ArraySetAsSeries(g_bufCycle24, true);

   IndicatorSetString(INDICATOR_SHORTNAME, "GPU WaveViz Solo");

   g_prevLogging = GpuLogsEnabled();
   GpuSetLogging(InpVerboseLog);

   if(!g_engine.Initialize(InpGPUDevice, InpFFTWindow, InpHop, 1, false))
     {
      Print("[WaveViz Solo] Falha ao inicializar a GPU Engine");
      return(INIT_FAILED);
     }

   BuildCyclePeriods();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   g_engine.Shutdown();
   GpuSetLogging(g_prevLogging);
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
   // Enquanto ainda não temos nenhum frame real da GPU,
   // apenas mostramos o progresso de inicialização (sem desenhar fallback enganoso).
   if(!g_hasWaveResult)
     {
      g_initProgress = (InpFFTWindow > 0 ? (double)rates_total / (double)InpFFTWindow : 0.0);
      if(g_initProgress < 0.0)
         g_initProgress = 0.0;
      if(g_initProgress > 1.0)
         g_initProgress = 1.0;

      if(g_initProgress < 1.0)
         g_statusText = StringFormat("Init WaveViz %.0f%% (%d/%d barras)",
                                     g_initProgress * 100.0,
                                     rates_total,
                                     InpFFTWindow);
      else
         g_statusText = "Init WaveViz: histórico ok, aguardando 1º frame GPU...";
     }

   Comment(g_statusText);

   if(g_job.active)
      FetchCurrentResult();

   if(!g_job.active && g_job.submitted_bars != rates_total)
     {
      if(SubmitJob(rates_total))
         FetchCurrentResult();
     }

   return(rates_total);
  }
