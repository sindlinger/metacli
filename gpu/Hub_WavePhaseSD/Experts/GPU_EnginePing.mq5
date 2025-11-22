//+------------------------------------------------------------------+
//| GPU_EnginePing.mq5                                              |
//| Teste mínimo de handshake com GpuEngineService.exe              |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "1.000"
#property strict

#include <GPU/GPU_Engine.mqh>

input int  InpGPU       = 0;
input int  InpWindow    = 1024;
input int  InpHop       = 256;
input int  InpBatch     = 8;
input bool InpProfiling = false;
input bool InpVerboseLog = true;

CGpuEngineClient g_engine;
bool              g_prevGpuLogging = true;

int OnInit()
  {
   g_prevGpuLogging = GpuLogsEnabled();
   GpuSetLogging(InpVerboseLog);
   Print("[Ping] Iniciando teste de conexão...");
   if(!g_engine.Initialize(InpGPU, InpWindow, InpHop, InpBatch, InpProfiling))
     {
      Print("[Ping] GpuEngine_Init falhou. Veja o log do serviço.");
      return INIT_FAILED;
     }

   double frames[];
   ArrayResize(frames, (int)InpWindow);
   ArrayInitialize(frames, 0.0);

   ulong handle = 0;
   if(!g_engine.SubmitJob(frames, 1, 12345, 0, handle))
     {
      Print("[Ping] SubmitJob falhou.");
      return INIT_FAILED;
     }

   int status = GPU_ENGINE_IN_PROGRESS;
   for(int i=0; i<20 && status == GPU_ENGINE_IN_PROGRESS; ++i)
     {
      g_engine.PollStatus(handle, status);
      Sleep(100);
     }

   if(status != GPU_ENGINE_READY)
     {
      PrintFormat("[Ping] PollStatus terminou com status=%d", status);
      return INIT_FAILED;
     }

   // Buffers de saída – dimensionados para o pior caso
   // (frame único + até 24 ciclos).
   const int frame_len   = InpWindow;
   const int max_cycles  = 24;
   const int total_frame = (frame_len > 0 ? frame_len : 1);
   const int total_cycle = total_frame * max_cycles;

   double wave[];
   double preview[];
   double cycles[];
   double noise[];
   double phase[];
   double phase_unwrapped[];
   double amplitude[];
   double period[];
   double frequency[];
   double eta[];
   double countdown[];
   double recon[];
   double kalman[];
   double confidence[];
   double amp_delta[];
   double turn_signal[];
   double direction[];
   double power[];
   double velocity[];
   double phase_all[];
   double phase_unwrapped_all[];
   double amplitude_all[];
   double period_all[];
   double frequency_all[];
   double eta_all[];
   double countdown_all[];
   double direction_all[];
   double recon_all[];
   double kalman_all[];
   double turn_all[];
   double confidence_all[];
   double amp_delta_all[];
   double power_all[];
   double velocity_all[];
   double plv_cycles[];
   double snr_cycles[];

   ArrayResize(wave,                 total_frame);
   ArrayResize(preview,              total_frame);
   ArrayResize(cycles,               total_cycle);
   ArrayResize(noise,                total_frame);
   ArrayResize(phase,                total_frame);
   ArrayResize(phase_unwrapped,      total_frame);
   ArrayResize(amplitude,            total_frame);
   ArrayResize(period,               total_frame);
   ArrayResize(frequency,            total_frame);
   ArrayResize(eta,                  total_frame);
   ArrayResize(countdown,            total_frame);
   ArrayResize(recon,                total_frame);
   ArrayResize(kalman,               total_frame);
   ArrayResize(confidence,           total_frame);
   ArrayResize(amp_delta,            total_frame);
   ArrayResize(turn_signal,          total_frame);
   ArrayResize(direction,            total_frame);
   ArrayResize(power,                total_frame);
   ArrayResize(velocity,             total_frame);
   ArrayResize(phase_all,            total_cycle);
   ArrayResize(phase_unwrapped_all,  total_cycle);
   ArrayResize(amplitude_all,        total_cycle);
   ArrayResize(period_all,           total_cycle);
   ArrayResize(frequency_all,        total_cycle);
   ArrayResize(eta_all,              total_cycle);
   ArrayResize(countdown_all,        total_cycle);
   ArrayResize(direction_all,        total_cycle);
   ArrayResize(recon_all,            total_cycle);
   ArrayResize(kalman_all,           total_cycle);
   ArrayResize(turn_all,             total_cycle);
   ArrayResize(confidence_all,       total_cycle);
   ArrayResize(amp_delta_all,        total_cycle);
   ArrayResize(power_all,            total_cycle);
   ArrayResize(velocity_all,         total_cycle);
   ArrayResize(plv_cycles,           max_cycles);
   ArrayResize(snr_cycles,           max_cycles);
   GpuEngineResultInfo info;

   if(!g_engine.FetchResult(handle,
                            wave,
                            preview,
                            cycles,
                            noise,
                            phase,
                            phase_unwrapped,
                            amplitude,
                            period,
                            frequency,
                            eta,
                            countdown,
                            recon,
                            kalman,
                            confidence,
                            amp_delta,
                            turn_signal,
                            direction,
                            power,
                            velocity,
                            phase_all,
                            phase_unwrapped_all,
                            amplitude_all,
                            period_all,
                            frequency_all,
                            eta_all,
                            countdown_all,
                            direction_all,
                            recon_all,
                            kalman_all,
                            turn_all,
                            confidence_all,
                            amp_delta_all,
                            power_all,
                            velocity_all,
                            plv_cycles,
                            snr_cycles,
                            info))
     {
      Print("[Ping] FetchResult falhou.");
      return INIT_FAILED;
     }

   PrintFormat("[Ping] Sucesso! frames=%d, cycles=%d, elapsed=%.2f ms",
               info.frame_count,
               info.cycle_count,
               info.elapsed_ms);

   EventSetTimer(1);
   return INIT_SUCCEEDED;
  }

void OnTimer()
  {
   EventKillTimer();
   ExpertRemove();
  }

void OnDeinit(const int reason)
  {
   g_engine.Shutdown();
   GpuSetLogging(g_prevGpuLogging);
   Print("[Ping] Finalizado.");
  }
