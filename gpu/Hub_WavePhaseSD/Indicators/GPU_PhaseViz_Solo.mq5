//+------------------------------------------------------------------+
//| GPU_PhaseViz_Solo.mq5                                           |
//| Visualizador autônomo de fase/amplitude via GPU Engine.         |
//| Submete um frame ao serviço e desenha os buffers resultantes.   |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "1.000"
#property strict

#property indicator_separate_window
#property indicator_buffers 12
#property indicator_plots   12

#property indicator_label1  "Phase(deg)"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrGold
#property indicator_width1  2

#property indicator_label2  "Amplitude"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDodgerBlue

#property indicator_label3  "Period(bars)"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrLime

#property indicator_label4  "ETA(bars)"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrOrange

#property indicator_label5  "Reconstructed"
#property indicator_type5   DRAW_LINE
#property indicator_color5  clrWhite

#property indicator_label6  "Confidence"
#property indicator_type6   DRAW_LINE
#property indicator_color6  clrPaleGreen

#property indicator_label7  "dAmp"
#property indicator_type7   DRAW_LINE
#property indicator_color7  clrMediumVioletRed

#property indicator_label8  "Phase Unwrapped"
#property indicator_type8   DRAW_LINE
#property indicator_color8  clrSlateBlue

#property indicator_label9  "Kalman"
#property indicator_type9   DRAW_LINE
#property indicator_color9  clrYellow
#property indicator_width9  2

#property indicator_label10 "TurnPulse"
#property indicator_type10  DRAW_HISTOGRAM
#property indicator_color10 clrOrangeRed
#property indicator_width10 2

#property indicator_label11 "Countdown"
#property indicator_type11  DRAW_LINE
#property indicator_color11 clrTomato

#property indicator_label12 "Direction"
#property indicator_type12  DRAW_LINE
#property indicator_color12 clrChocolate

#include <GPU/GPU_Engine.mqh>

enum SoloFeedMode
  {
   Feed_Close = 0
  };

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

input KalmanPresetOption InpKalmanPreset         = KalmanBalanced;
input double             InpKalmanProcessNoise   = 1.0e-4;
input double             InpKalmanMeasurementNoise = 2.5e-3;
input double             InpKalmanInitVariance   = 0.5;
input double             InpKalmanPlvThreshold   = 0.35;
input int                InpKalmanMaxIterations  = 48;
input double             InpKalmanConvergenceEps = 1.0e-4;

input SoloFeedMode InpFeedMode        = Feed_Close;

const uint JOB_FLAG_STFT   = 1;
const uint JOB_FLAG_CYCLES = 2;

double g_bufPhase[];
double g_bufAmplitude[];
double g_bufPeriod[];
double g_bufEta[];
double g_bufRecon[];
double g_bufConfidence[];
double g_bufAmpDelta[];
double g_bufPhaseUnwrapped[];
double g_bufKalman[];
double g_bufTurn[];
double g_bufCountdown[];
double g_bufDirection[];

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
double g_directionOut[];
double g_turnOut[];

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


CGpuEngineClient g_engine;
bool              g_prevLogging   = true;

struct PendingJob
  {
   bool  active;
   ulong handle;
   ulong tag;
   int   submitted_bars;
  };

PendingJob g_job = { false, 0, 0, -1 };

GpuEngineResultInfo g_lastInfo;

bool EnsureCapacity(double &buffer[], const int required, const string name)
  {
   if(required <= 0)
      return true;
   const int current = ArraySize(buffer);
   if(current >= required)
      return true;
   int result = ArrayResize(buffer, required);
   if(result == -1)
     {
      if(InpVerboseLog)
         PrintFormat("[PhaseViz Solo] Falha ao redimensionar %s para %d", name, required);
      return false;
     }
   return true;
  }

void ClearBuffers()
  {
   ArrayInitialize(g_bufPhase,          EMPTY_VALUE);
   ArrayInitialize(g_bufAmplitude,      EMPTY_VALUE);
   ArrayInitialize(g_bufPeriod,         EMPTY_VALUE);
   ArrayInitialize(g_bufEta,            EMPTY_VALUE);
   ArrayInitialize(g_bufRecon,          EMPTY_VALUE);
   ArrayInitialize(g_bufConfidence,     EMPTY_VALUE);
   ArrayInitialize(g_bufAmpDelta,       EMPTY_VALUE);
   ArrayInitialize(g_bufPhaseUnwrapped, EMPTY_VALUE);
   ArrayInitialize(g_bufKalman,         EMPTY_VALUE);
   ArrayInitialize(g_bufTurn,           EMPTY_VALUE);
   ArrayInitialize(g_bufCountdown,      EMPTY_VALUE);
   ArrayInitialize(g_bufDirection,      EMPTY_VALUE);
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
   const int hop = (InpHop > 0 ? InpHop : 1);
   const int estimated_frames = MathMax(1, (safe_frame_total + hop - 1) / hop);
   const int frame_required   = safe_frame_total * estimated_frames;
   const int cycle_required   = safe_cycle_total * estimated_frames;
   const int cycle_list_required = safe_cycle_count * estimated_frames;

   bool ok = true;
   ok = ok && EnsureCapacity(g_waveOut,     frame_required,   "wave_out");
   ok = ok && EnsureCapacity(g_previewOut,  frame_required,   "preview_out");
   ok = ok && EnsureCapacity(g_cyclesOut,   cycle_required,   "cycles_out");
   ok = ok && EnsureCapacity(g_noiseOut,    frame_required,   "noise_out");

   ok = ok && EnsureCapacity(g_phaseOut,          frame_required, "phase_out");
   ok = ok && EnsureCapacity(g_phaseUnwrappedOut, frame_required, "phase_unwrapped_out");
   ok = ok && EnsureCapacity(g_amplitudeOut,      frame_required, "amplitude_out");
   ok = ok && EnsureCapacity(g_periodOut,         frame_required, "period_out");
   ok = ok && EnsureCapacity(g_frequencyOut,      frame_required, "frequency_out");
   ok = ok && EnsureCapacity(g_etaOut,            frame_required, "eta_out");
   ok = ok && EnsureCapacity(g_countdownOut,      frame_required, "countdown_out");
   ok = ok && EnsureCapacity(g_reconOut,          frame_required, "recon_out");
   ok = ok && EnsureCapacity(g_kalmanOut,         frame_required, "kalman_out");
   ok = ok && EnsureCapacity(g_confidenceOut,     frame_required, "confidence_out");
   ok = ok && EnsureCapacity(g_ampDeltaOut,       frame_required, "amp_delta_out");
   ok = ok && EnsureCapacity(g_directionOut,      frame_required, "direction_out");
   ok = ok && EnsureCapacity(g_turnOut,           frame_required, "turn_out");
   ok = ok && EnsureCapacity(g_powerOut,          frame_required, "power_out");
   ok = ok && EnsureCapacity(g_velocityOut,       frame_required, "velocity_out");

   ok = ok && EnsureCapacity(g_phaseAllOut,          cycle_required, "phase_all_out");
   ok = ok && EnsureCapacity(g_phaseUnwrappedAllOut, cycle_required, "phase_unwrapped_all_out");
   ok = ok && EnsureCapacity(g_amplitudeAllOut,      cycle_required, "amplitude_all_out");
   ok = ok && EnsureCapacity(g_periodAllOut,         cycle_required, "period_all_out");
   ok = ok && EnsureCapacity(g_frequencyAllOut,      cycle_required, "frequency_all_out");
   ok = ok && EnsureCapacity(g_etaAllOut,            cycle_required, "eta_all_out");
   ok = ok && EnsureCapacity(g_countdownAllOut,      cycle_required, "countdown_all_out");
   ok = ok && EnsureCapacity(g_directionAllOut,      cycle_required, "direction_all_out");
   ok = ok && EnsureCapacity(g_reconAllOut,          cycle_required, "recon_all_out");
   ok = ok && EnsureCapacity(g_kalmanAllOut,         cycle_required, "kalman_all_out");
   ok = ok && EnsureCapacity(g_turnAllOut,           cycle_required, "turn_all_out");
   ok = ok && EnsureCapacity(g_confidenceAllOut,     cycle_required, "confidence_all_out");
   ok = ok && EnsureCapacity(g_ampDeltaAllOut,       cycle_required, "amp_delta_all_out");
   ok = ok && EnsureCapacity(g_powerAllOut,          cycle_required, "power_all_out");
   ok = ok && EnsureCapacity(g_velocityAllOut,       cycle_required, "velocity_all_out");

   ok = ok && EnsureCapacity(g_plvCyclesOut, cycle_list_required, "plv_cycles_out");
   ok = ok && EnsureCapacity(g_snrCyclesOut, cycle_list_required, "snr_cycles_out");

   if(!ok && InpVerboseLog)
      Print("[PhaseViz Solo] Aviso: redimensionamento de buffers falhou");
  }

void CopyResultsToBuffers(const GpuEngineResultInfo &info)
  {
   const int frame_length = info.frame_length;
   const int frame_count  = info.frame_count;
   if(frame_length <= 0 || frame_count <= 0)
      return;

   ClearBuffers();

   const int latest_offset = (frame_count - 1) * frame_length;

   for(int i=0; i<frame_length; ++i)
     {
      const int src = latest_offset + (frame_length - 1 - i);
      const int dest= i;

      g_bufPhase[dest]          = g_phaseOut[src];
      g_bufAmplitude[dest]      = g_amplitudeOut[src];
      g_bufPeriod[dest]         = g_periodOut[src];
      g_bufEta[dest]            = g_etaOut[src];
      g_bufCountdown[dest]      = g_countdownOut[src];
      g_bufRecon[dest]          = g_reconOut[src];
      g_bufConfidence[dest]     = g_confidenceOut[src];
      g_bufAmpDelta[dest]       = g_ampDeltaOut[src];
      g_bufPhaseUnwrapped[dest] = g_phaseUnwrappedOut[src];
      g_bufKalman[dest]         = g_kalmanOut[src];
      g_bufTurn[dest]           = g_turnOut[src];
      double dir = (ArraySize(g_directionOut) > src ? g_directionOut[src] : (g_countdownOut[src] >= 0.0 ? 1.0 : -1.0));
      g_bufDirection[dest]      = dir;
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
   SetIndexBuffer(0, g_bufPhase,         INDICATOR_DATA);
   SetIndexBuffer(1, g_bufAmplitude,     INDICATOR_DATA);
   SetIndexBuffer(2, g_bufPeriod,        INDICATOR_DATA);
   SetIndexBuffer(3, g_bufEta,           INDICATOR_DATA);
   SetIndexBuffer(4, g_bufRecon,         INDICATOR_DATA);
   SetIndexBuffer(5, g_bufConfidence,    INDICATOR_DATA);
   SetIndexBuffer(6, g_bufAmpDelta,      INDICATOR_DATA);
   SetIndexBuffer(7, g_bufPhaseUnwrapped,INDICATOR_DATA);
   SetIndexBuffer(8, g_bufKalman,        INDICATOR_DATA);
   SetIndexBuffer(9, g_bufTurn,          INDICATOR_DATA);
   SetIndexBuffer(10,g_bufCountdown,     INDICATOR_DATA);
   SetIndexBuffer(11,g_bufDirection,     INDICATOR_DATA);

   ArraySetAsSeries(g_bufPhase,         true);
   ArraySetAsSeries(g_bufAmplitude,     true);
   ArraySetAsSeries(g_bufPeriod,        true);
   ArraySetAsSeries(g_bufEta,           true);
   ArraySetAsSeries(g_bufRecon,         true);
   ArraySetAsSeries(g_bufConfidence,    true);
   ArraySetAsSeries(g_bufAmpDelta,      true);
   ArraySetAsSeries(g_bufPhaseUnwrapped,true);
   ArraySetAsSeries(g_bufKalman,        true);
   ArraySetAsSeries(g_bufTurn,          true);
   ArraySetAsSeries(g_bufCountdown,     true);
   ArraySetAsSeries(g_bufDirection,     true);

   IndicatorSetString(INDICATOR_SHORTNAME, "GPU PhaseViz Solo");

   g_prevLogging = GpuLogsEnabled();
   GpuSetLogging(InpVerboseLog);

   if(!g_engine.Initialize(InpGPUDevice, InpFFTWindow, InpHop, 1, false))
     {
      Print("[PhaseViz Solo] Falha ao inicializar a GPU Engine");
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
