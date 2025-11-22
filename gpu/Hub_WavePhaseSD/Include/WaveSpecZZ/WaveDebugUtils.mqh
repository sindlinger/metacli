//+------------------------------------------------------------------+
//| WaveDebugUtils.mqh                                               |
//| Utilitários para exportar buffers internos do WaveSpecZZ.        |
//+------------------------------------------------------------------+
#ifndef __WAVE_DEBUG_UTILS_MQH__
#define __WAVE_DEBUG_UTILS_MQH__

namespace WaveDebug
{
//+------------------------------------------------------------------+
//| Salva séries relevantes em CSV (1 linha por índice).            |
//| Retorna true em caso de sucesso.                                |
//+------------------------------------------------------------------+
bool DumpSeriesCSV(const string filename,
                   const double &zigzag_raw[],
                   const double &gaussian[],
                   const double &zigzag_type[],
                   const int signal_len,
                   const double &src_fft_real[],
                   const double &src_fft_imag[],
                   const int fft_len,
                   const double &src_spectrum[],
                   const double &src_spectrum_masked[],
                   const int spectrum_len)
  {
   int handle = FileOpen(filename, FILE_WRITE | FILE_CSV, ';');
   if(handle == INVALID_HANDLE)
     {
      int err = GetLastError();
      PrintFormat("WaveDebug: falha ao abrir %s (erro=%d)", filename, err);
      return false;
     }

   FileWrite(handle, "index", "zigzag_raw", "gaussian",
             "zigzag_type", "fft_real", "fft_imag",
             "spectrum", "spectrum_masked");

   int max_rows = signal_len;
   if(fft_len > max_rows) max_rows = fft_len;
   if(spectrum_len > max_rows) max_rows = spectrum_len;

   for(int i = 0; i < max_rows; ++i)
     {
      double zig = (i < signal_len) ? zigzag_raw[i] : 0.0;
      double gauss = (i < signal_len) ? gaussian[i] : 0.0;
      double ztype = (i < signal_len) ? zigzag_type[i] : 0.0;
      double real_val = (i < fft_len) ? src_fft_real[i] : 0.0;
      double imag_val = (i < fft_len) ? src_fft_imag[i] : 0.0;
      double spec_val = (i < spectrum_len) ? src_spectrum[i] : 0.0;
      double spec_mask_val = (i < spectrum_len) ? src_spectrum_masked[i] : 0.0;

      FileWrite(handle, i, zig, gauss, ztype,
                real_val, imag_val, spec_val, spec_mask_val);
     }

   FileClose(handle);
   PrintFormat("WaveDebug: dump salvo em %s", filename);
   return true;
  }

//+------------------------------------------------------------------+
//| Dumpa métricas completas do PhaseViz (fase/frequência/countdown) |
//+------------------------------------------------------------------+
bool DumpPhaseVizSeriesCSV(const string filename,
                           const double &zigzag_raw[],
                           const double &gaussian[],
                           const double &zigzag_type[],
                           const double &phase_amp[],
                           const double &phase_wrapped[],
                           const double &phase_unwrapped[],
                           const double &phase_frequency[],
                           const double &phase_period[],
                           const double &countdown_bars[],
                           const int length,
                           const bool normalize_amplitude,
                           const double amplitude_scale)
  {
   int handle = FileOpen(filename, FILE_WRITE | FILE_CSV, ';');
   if(handle == INVALID_HANDLE)
     {
      int err = GetLastError();
      PrintFormat("WaveDebug: falha ao abrir %s (erro=%d)", filename, err);
      return false;
     }

   FileWrite(handle,
             "index",
             "zigzag_raw",
             "gaussian",
             "zigzag_type",
             "phase_wrapped_deg",
             "phase_unwrapped_deg",
             "frequency_cycles_per_bar",
             "period_bars",
             "amplitude",
             "countdown_bars");

   double norm = 1.0;
   if(normalize_amplitude)
     {
      int idx = ArrayMaximum(phase_amp, 0, length);
      if(idx >= 0)
        {
         double max_val = MathAbs(phase_amp[idx]);
         if(max_val > 1e-12)
            norm = max_val;
        }
     }

   double amp_scale = (MathAbs(amplitude_scale) > 1e-12) ? amplitude_scale : 1.0;
   const double rad_to_deg = 180.0 / M_PI;

   for(int i = 0; i < length; ++i)
     {
      double zig = zigzag_raw[i];
      double gauss = gaussian[i];
      double zig_type = zigzag_type[i];

      double amp = phase_amp[i];
      if(normalize_amplitude && norm > 1e-12)
         amp /= norm;
      amp /= amp_scale;

      double phase_wr_deg = phase_wrapped[i] * rad_to_deg;
      double phase_un_deg = phase_unwrapped[i] * rad_to_deg;
      double freq = phase_frequency[i];
      double period = phase_period[i];
      double countdown = countdown_bars[i];

      if(!MathIsValidNumber(phase_wr_deg))
         phase_wr_deg = EMPTY_VALUE;
      if(!MathIsValidNumber(phase_un_deg))
         phase_un_deg = EMPTY_VALUE;
      if(!MathIsValidNumber(freq))
         freq = EMPTY_VALUE;
      if(!MathIsValidNumber(period))
         period = EMPTY_VALUE;
      if(!MathIsValidNumber(amp))
         amp = EMPTY_VALUE;
      if(!MathIsValidNumber(countdown))
         countdown = EMPTY_VALUE;

      FileWrite(handle,
                i,
                zig,
                gauss,
                zig_type,
                phase_wr_deg,
                phase_un_deg,
                freq,
                period,
                amp,
                countdown);
     }

   FileClose(handle);
   PrintFormat("WaveDebug: PhaseViz dump salvo em %s", filename);
   return true;
  }
} // namespace WaveDebug

#endif // __WAVE_DEBUG_UTILS_MQH__
