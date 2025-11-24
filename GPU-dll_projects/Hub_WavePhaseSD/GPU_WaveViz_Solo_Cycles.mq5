//+------------------------------------------------------------------+
//| GPU_WaveViz_Solo_Cycles.mq5                                      |
//| Visualizador autônomo dos ciclos dominantes via GPU Engine.      |
//| Cada instância possui parâmetros próprios de ciclos/Kalman.      |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "1.100"
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

#include <GPU/GPU_Engine.mqh>

enum SoloFeedMode
  {
   Feed_Close = 0
  };

// Parâmetros independentes para este indicador
input bool         InpVerboseLog      = true;
input int          InpGPUDevice       = 0;
input int          InpFFTWindow       = 4096;
input int          InpHop             = 1024;
input bool         InpUseManualCycles = false;
input int          InpCycleCount      = 24;
input double       InpCycleMinPeriod  = 18.0;
input double       InpCycleMaxPeriod  = 1440.0;
input double       InpCycleWidth      = 0.25;
input int          InpMaxCandidates   = 24;
input double       InpGaussSigmaPeriod       = 48.0;
input double       InpMaskThreshold   = 0.05;
input double       InpMaskSoftness    = 0.20;
input double       InpMaskMinPeriod   = 18.0;
input double       InpMaskMaxPeriod   = 512.0;

enum KalmanPresetOption
  {
   KalmanSmooth   = 0,
   KalmanBalanced = 1,
   KalmanReactive = 2,
   KalmanManual   = 3
  };

// Kalman pode influenciar a máscara/seleção de ciclos
input KalmanPresetOption InpKalmanPreset         = KalmanBalanced;
input double             InpKalmanProcessNoise   = 1.0e-4;
input double             InpKalmanMeasurementNoise = 2.5e-3;
input double             InpKalmanInitVariance   = 0.5;
input double             InpKalmanPlvThreshold   = 0.35;
input int                InpKalmanMaxIterations  = 48;
input double             InpKalmanConvergenceEps = 1.0e-4;

input SoloFeedMode InpFeedMode       = Feed_Close;

const uint JOB_FLAG_STFT   = 1;
const uint JOB_FLAG_CYCLES = 2;

double g_bufCycle1[];
double g_bufCycle2[];
double g_bufCycle3[];
double g_bufCycle4[];

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

CGpuEngineClient g_engine;
bool             g_prevLogging   = true;

struct PendingJob
  {
   bool  active;
   ulong handle;
   ulong tag;
   int   submitted_bars;
  };

PendingJob g_job = { false, 0, 0, -1 };

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

void ClearCycleBuffers()
  {
   ArrayInitialize(g_bufCycle1, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle2, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle3, EMPTY_VALUE);
   ArrayInitialize(g_bufCycle4, EMPTY_VALUE);
  }

void SetCycleValue(const int index,
                   const int bar_index,
                   const double value)
  {
   switch(index)
     {
      case 0: g_bufCycle1[bar_index] = value; break;
      case 1: g_bufCycle2[bar_index] = value; break;
      case 2: g_bufCycle3[bar_index] = value; break;
      case 3: g_bufCycle4[bar_index] = value; break;
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
   const ulong tag    = ++g_job.tag;

   int cycle_count = (int)MathMin((double)ArraySize(g_cyclePeriods), 24.0);
   if(cycle_count < 0)
      cycle_count = 0;

   uint flags  = JOB_FLAG_STFT;
   bool submitted;
   if(cycle_count > 0)
     {
      flags |= JOB_FLAG_CYCLES;
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

   g_job.active         = true;
   g_job.handle         = handle;
   g_job.submitted_bars = rates_total;
   return(true);
  }

void EnsureFetchBuffers()
  {
   const int frame_total = InpFFTWindow;
   const bool use_manual_cycles = InpUseManualCycles && ArraySize(g_cyclePeriods) > 0;
   const int max_cycles = (use_manual_cycles ? (int)MathMin((double)ArraySize(g_cyclePeriods), 24.0)
                                            : (int)MathMax(MathMin((double)InpMaxCandidates, 24.0), 0.0));
   const int safe_frame_total = (frame_total > 0 ? frame_total : 1);
   const int safe_cycles      = (max_cycles > 0 ? max_cycles : 0);
   const int safe_cycle_total = (safe_cycles > 0 ? safe_cycles * safe_frame_total : 1);
   const int safe_cycle_count = (max_cycles > 0 ? max_cycles : 1);

   ArrayResize(g_waveOut,     safe_frame_total);
   ArrayResize(g_previewOut,  safe_frame_total);
   ArrayResize(g_cyclesOut,   safe_cycle_total);
   ArrayResize(g_noiseOut,    safe_frame_total);

   ArrayResize(g_phaseOut,          safe_frame_total);
   ArrayResize(g_phaseUnwrappedOut, safe_frame_total);
   ArrayResize(g_amplitudeOut,      safe_frame_total);
   ArrayResize(g_periodOut,         safe_frame_total);
   ArrayResize(g_frequencyOut,      safe_frame_total);
   ArrayResize(g_etaOut,            safe_frame_total);
   ArrayResize(g_countdownOut,      safe_frame_total);
   ArrayResize(g_reconOut,          safe_frame_total);
   ArrayResize(g_kalmanOut,         safe_frame_total);
   ArrayResize(g_confidenceOut,     safe_frame_total);
   ArrayResize(g_ampDeltaOut,       safe_frame_total);
   ArrayResize(g_turnOut,           safe_frame_total);
   ArrayResize(g_directionOut,      safe_frame_total);
   ArrayResize(g_powerOut,          safe_frame_total);
   ArrayResize(g_velocityOut,       safe_frame_total);

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
   const int total_cycle_count = info.cycle_count;

   if(frame_length <= 0 || frame_count <= 0)
      return;

   ClearCycleBuffers();

   const int samples_total = frame_length;
   const int total_span    = frame_length * frame_count;
   const int frame_offset  = (frame_count - 1) * frame_length;

   const int max_visual_cycles = 4;
   const int target_cycle_count = (int)MathMin((double)max_visual_cycles, (double)total_cycle_count);

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

   for(int i=0; i<samples_total; ++i)
     {
      const int src_index  = frame_offset + i;
      const int dest_index = i;

      const int needed = total_span * total_cycle_count;
      bool has_recon_all  = (ArraySize(g_reconAllOut) >= needed);
      bool has_cycles_all = (ArraySize(g_cyclesOut)   >= needed);

      if(selected_count > 0)
        {
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

   if(InpVerboseLog)
     {
      const double c1_last = (samples_total > 0 ? g_bufCycle1[0] : 0.0);
      PrintFormat("[WaveViz Cycles][DBG] frame_len=%d cycles=%d cycle1_first=%.6f",
                  samples_total,
                  selected_count,
                  c1_last);
     }
  }

bool FetchCurrentResult()
  {
   int status = GPU_ENGINE_IN_PROGRESS;
   if(g_engine.PollStatus(g_job.handle, status) != GPU_ENGINE_OK)
      return(false);

   if(status != GPU_ENGINE_READY)
      return(false);

   EnsureFetchBuffers();

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
   SetIndexBuffer(0, g_bufCycle1, INDICATOR_DATA);
   SetIndexBuffer(1, g_bufCycle2, INDICATOR_DATA);
   SetIndexBuffer(2, g_bufCycle3, INDICATOR_DATA);
   SetIndexBuffer(3, g_bufCycle4, INDICATOR_DATA);

   ArraySetAsSeries(g_bufCycle1, true);
   ArraySetAsSeries(g_bufCycle2, true);
   ArraySetAsSeries(g_bufCycle3, true);
   ArraySetAsSeries(g_bufCycle4, true);

   IndicatorSetString(INDICATOR_SHORTNAME, "GPU WaveViz Solo - Cycles");

   g_prevLogging = GpuLogsEnabled();
   GpuSetLogging(InpVerboseLog);

   if(!g_engine.Initialize(InpGPUDevice, InpFFTWindow, InpHop, 1, false))
     {
      Print("[WaveViz Solo Cycles] Falha ao inicializar a GPU Engine");
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
   if(rates_total < InpFFTWindow)
      return(prev_calculated);

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

