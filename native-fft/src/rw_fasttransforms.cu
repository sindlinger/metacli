// fasttransforms CUDA implementation for ReversalWave
// Exports two functions expected by ReversalWaveBridge.mqh:
//  - gpu_reversal_wave_process
//  - gpu_reversal_wave_synthetic_test
//
// Notes:
//  * Built for Windows 64-bit, __stdcall calling convention.
//  * Uses cuFFT (double precision) for R2C/C2R pipeline.
//  * Keeps the contract defined in mql5/Include/ReversalWaveBridge.mqh.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <mutex>
#include <cstdarg>
#include <cstdio>

#ifndef _WIN32
#define __declspec(x) __attribute__((visibility("default")))
#define __stdcall
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

static const char* kLogPath = "MQL5/Files/reversal_wave_debug.log";

// ---------------- Logging helpers ----------------

static std::mutex g_logMutex;

static void rw_log(const char* fmt, ...) {
  std::lock_guard<std::mutex> lock(g_logMutex);
  FILE* f = fopen(kLogPath, "a");
  if (!f) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  fprintf(f, "\n");
  va_end(ap);
  fclose(f);
}

static void rw_log_array_sample(const char* name, const double* host, int n, bool mt5Series) {
  if (!host || n <= 0) return;
  int end = n - 1;
  int a0 = 0, a1 = std::min(1, end), a2 = std::min(2, end);
  int b0 = std::max(0, end - 2), b1 = std::max(0, end - 1), b2 = end;
  rw_log("[%s] order=%s size=%d sample(head)=[%.6f, %.6f, %.6f] sample(tail)=[%.6f, %.6f, %.6f]", name,
         mt5Series ? "MT5_series(latest=0)" : "chronological(oldest=0)", n,
         host[a0], host[a1], host[a2], host[b0], host[b1], host[b2]);
}

// ---------------- CUDA helpers ----------------

inline bool checkCuda(cudaError_t st) { return st == cudaSuccess; }
inline bool checkCufft(cufftResult st) { return st == CUFFT_SUCCESS; }

__global__ void k_reverse(const double* src, double* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[n - 1 - i];
  if (blockIdx.x == 0 && threadIdx.x < 2 && i < n) {
    printf("k_reverse idx=%d src=%f dst=%f\n", i, src[i], dst[i]);
  }
}

__global__ void k_mean_reduce(const double* x, double* partial, int n) {
  extern __shared__ double s[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double v = (idx < n) ? x[idx] : 0.0;
  s[threadIdx.x] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

__global__ void k_sq_reduce(const double* x, double* partial, int n) {
  extern __shared__ double s[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double v = (idx < n) ? x[idx] * x[idx] : 0.0;
  s[threadIdx.x] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

__global__ void k_sum_x_tx(const double* x, double* partial_x, double* partial_tx, int n) {
  extern __shared__ double s[];
  double* sx = s;
  double* stx = s + blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double v = (idx < n) ? x[idx] : 0.0;
  sx[threadIdx.x] = v;
  stx[threadIdx.x] = (idx < n) ? v * idx : 0.0;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx[threadIdx.x] += sx[threadIdx.x + stride];
      stx[threadIdx.x] += stx[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_x[blockIdx.x] = sx[0];
    partial_tx[blockIdx.x] = stx[0];
  }
}

__global__ void k_center(double* x, double mean, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] -= mean;
  if (blockIdx.x == 0 && threadIdx.x == 0)
    printf("k_center mean=%f first=%f\n", mean, x[0]);
}

__global__ void k_hann(double* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double w = 0.5 * (1.0 - cos(2.0 * M_PI * i / (n - 1)));
    x[i] *= w;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0)
    printf("k_hann applied n=%d\n", n);
}

__global__ void k_norm_std(double* x, int n, double invStd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= invStd;
}

__global__ void k_detrend_ab(double* x, int n, double a, double b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = x[i] - (a + b * i);
  }
}

// Blackman window moving-mean detrend (time-domain, single pass)
__global__ void k_blackman_detrend(const double* src, double* dst, int n, int win) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || win <= 1) return;
  int radius = win / 2;
  int start = i - radius;
  int end   = i + radius;
  if (start < 0) start = 0;
  if (end >= n) end = n - 1;
  int len = end - start + 1;
  double acc = 0.0;
  double wsum = 0.0;
  for (int j = start; j <= end; ++j) {
    int k = j - start;
    double w = 0.42 - 0.5 * cos(2.0 * M_PI * k / (len - 1 + 1e-12)) + 0.08 * cos(4.0 * M_PI * k / (len - 1 + 1e-12));
    acc += src[j] * w;
    wsum += w;
  }
  if (wsum < 1e-12) wsum = 1e-12;
  double mean = acc / wsum;
  dst[i] = src[i] - mean;
}

// Replace NaN/Inf with 0 to keep the pipeline stable
__global__ void k_sanitize(double* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double v = x[i];
    if (!isfinite(v)) v = 0.0;
    x[i] = v;
  }
}

// candle 1-2-3 com pavio/corpo em GPU
__global__ void k_candle123(const double* open, const double* close, const double* high, const double* low,
                            double* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= 0 || i >= n-1) return;
  double o = open[i], c = close[i], hi = high[i], lo = low[i];
  double body = fabs(c - o);
  double upper = hi - fmax(o, c);
  double lower = fmin(o, c) - lo;
  double range = hi - lo;
  if (range < 1e-12) range = 1e-12;
  const double wickRatio = 0.5;
  const double rangeFrac = 0.25;
  bool upperOK = upper >= range * rangeFrac && upper >= body * wickRatio;
  bool lowerOK = lower >= range * rangeFrac && lower >= body * wickRatio;

  bool isTop = (hi > high[i-1]) && (hi > high[i+1]) &&
               (lo >= low[i-1]) && (lo >= low[i+1]) &&
               upperOK;
  bool isBottom = (lo < low[i-1]) && (lo < low[i+1]) &&
                  (hi <= high[i-1]) && (hi <= high[i+1]) &&
                  lowerOK;
  // Doc: +1 topo, -1 fundo
  if (isTop) out[i] = 1.0;
  else if (isBottom) out[i] = -1.0;
  else out[i] = 0.0;
}

// zigzag step-hold do valor do pivô
__global__ void k_step_hold(const double* pivots, double* out, int n) {
  double last = 0.0;
  for (int i = 0; i < n; ++i) {
    double v = pivots[i];
    if (fabs(v) > 1e-12) last = v;
    out[i] = last;
  }
}

// volume zscore janela fixa (O(20) por thread, aceitável)
__global__ void k_vol_zscore(const double* vol, double* out, int n, int win, double scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int start = i - win + 1;
  if (start < 0) start = 0;
  int len = i - start + 1;
  double sum = 0.0;
  for (int j = start; j <= i; ++j) sum += vol[j];
  double mean = sum / len;
  double sq = 0.0;
  for (int j = start; j <= i; ++j) {
    double d = vol[j] - mean;
    sq += d * d;
  }
  double sd = (sq <= 0.0) ? 1e-9 : sqrt(sq / len);
  out[i] = scale * (vol[i] - mean) / sd;
}

__global__ void k_blend(const double* price, const double* volume, const double* pivots,
                        double* out, int n, int mode, double pw, double vw, double zw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double p = price[i];
    double v = volume[i];
    double z = pivots[i];
    if (mode & RW_MODE_EMPHASIZE_PIVOT) z *= 1.5;
    out[i] = pw * p + vw * v + zw * z;
  }
}

// Simple band-pass: zero DC; optionally widen nothing else (placeholder for EXTENDED_BAND flag).
__global__ void k_bandpass(cufftDoubleComplex* f, int nComplex, int lowBin, int highBin) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nComplex) {
    if (i < lowBin || i > highBin) {
      f[i].x = 0.0;
      f[i].y = 0.0;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0)
    printf("k_bandpass n=%d low=%d high=%d\n", nComplex, lowBin, highBin);
}

__global__ void k_normalize(double* x, int n, double invN) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= invN;
}

// ---------------- Host utility ----------------

static double device_mean(double* d_x, int n) {
  const int threads = 256;
  int blocks = (n + threads - 1) / threads;
  std::vector<double> partial(blocks);
  double* d_partial = nullptr;
  cudaMalloc(&d_partial, blocks * sizeof(double));
  k_mean_reduce<<<blocks, threads, threads * sizeof(double)>>>(d_x, d_partial, n);
  cudaMemcpy(partial.data(), d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_partial);
  double sum = 0.0;
  for (double v : partial) sum += v;
  return sum / static_cast<double>(n);
}

static double device_std(double* d_x, int n, double mean) {
  const int threads = 256;
  int blocks = (n + threads - 1) / threads;
  std::vector<double> partial(blocks);
  double* d_partial = nullptr;
  cudaMalloc(&d_partial, blocks * sizeof(double));
  k_sq_reduce<<<blocks, threads, threads * sizeof(double)>>>(d_x, d_partial, n);
  cudaMemcpy(partial.data(), d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_partial);
  double sumsq = 0.0;
  for (double v : partial) sumsq += v;
  double var = sumsq / static_cast<double>(n) - mean * mean;
  if (var < 1e-18) var = 1e-18;
  return std::sqrt(var);
}

static void device_sums_xtx(double* d_x, int n, double& sum_x, double& sum_tx) {
  const int threads = 256;
  int blocks = (n + threads - 1) / threads;
  std::vector<double> partial_x(blocks);
  std::vector<double> partial_tx(blocks);
  double *d_px = nullptr, *d_ptx = nullptr;
  cudaMalloc(&d_px, blocks * sizeof(double));
  cudaMalloc(&d_ptx, blocks * sizeof(double));
  size_t shmem = threads * 2 * sizeof(double);
  k_sum_x_tx<<<blocks, threads, shmem>>>(d_x, d_px, d_ptx, n);
  cudaMemcpy(partial_x.data(), d_px, blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(partial_tx.data(), d_ptx, blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_px);
  cudaFree(d_ptx);
  sum_x = 0.0;
  sum_tx = 0.0;
  for (int i = 0; i < blocks; ++i) { sum_x += partial_x[i]; sum_tx += partial_tx[i]; }
}

// ---------------- API implementation ----------------

extern "C" {

__declspec(dllexport) int __stdcall gpu_reversal_wave_process(
    const double* price, const double* volume, const double* pivots,
    int length, int window, int modeFlags,
    double priceWeight, double volumeWeight, double pivotWeight,
    double* outWave, double* outConfidence, int* outFlags) {

  if (!price || !volume || !pivots || !outWave || !outConfidence || !outFlags ||
      length <= 0 || window <= 0)
    return RW_RESULT_INVALID_ARGUMENT;
  if (length < window)
    return RW_RESULT_NOT_ENOUGH_DATA;

  const int N = length;
  const int threads = 256;
  int blocks = (N + threads - 1) / threads;

  double *d_price = nullptr, *d_volume = nullptr, *d_pivots = nullptr;
  double *d_blend = nullptr, *d_wave = nullptr;
  cufftDoubleComplex* d_freq = nullptr;
  cufftHandle plan = 0;

  size_t realBytes = sizeof(double) * N;
  size_t freqBytes = sizeof(cufftDoubleComplex) * (N / 2 + 1);

  auto cleanup = [&]() {
    if (plan) { cufftDestroy(plan); plan = 0; }
    if (d_price) cudaFree(d_price);
    if (d_volume) cudaFree(d_volume);
    if (d_pivots) cudaFree(d_pivots);
    if (d_blend) cudaFree(d_blend);
    if (d_wave) cudaFree(d_wave);
    if (d_freq) cudaFree(d_freq);
  };

  if (!checkCuda(cudaMalloc(&d_price, realBytes))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  if (!checkCuda(cudaMalloc(&d_volume, realBytes))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  if (!checkCuda(cudaMalloc(&d_pivots, realBytes))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  if (!checkCuda(cudaMalloc(&d_blend, realBytes))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  if (!checkCuda(cudaMalloc(&d_wave, realBytes))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  if (!checkCuda(cudaMalloc(&d_freq, freqBytes))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }

  // Copy and reverse series (MT5 index 0 = latest).
  k_reverse<<<blocks, threads>>>(price, d_price, N);
  k_reverse<<<blocks, threads>>>(volume, d_volume, N);
  k_reverse<<<blocks, threads>>>(pivots, d_pivots, N);
  
  // Sanitize NaN/Inf
  k_sanitize<<<blocks, threads>>>(d_price, N);
  k_sanitize<<<blocks, threads>>>(d_volume, N);
  k_sanitize<<<blocks, threads>>>(d_pivots, N);

  rw_log("[gpu_reversal_wave_process] length=%d window=%d mode=%d weights(p/v/z)=%.3f/%.3f/%.3f", N, window, modeFlags, priceWeight, volumeWeight, pivotWeight);
  std::vector<double> tmp(N);
  cudaMemcpy(tmp.data(), d_price, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("price_chrono_after_reverse", tmp.data(), N, false);
  cudaMemcpy(tmp.data(), d_volume, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("volume_chrono_after_reverse", tmp.data(), N, false);
  cudaMemcpy(tmp.data(), d_pivots, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("pivots_chrono_after_reverse", tmp.data(), N, false);

  // Normalize price and volume to unit variance to avoid dominance.
  double meanV = device_mean(d_volume, N);
  double stdP = device_std(d_price, N, device_mean(d_price, N));
  double stdV = device_std(d_volume, N, meanV);
  k_center<<<blocks, threads>>>(d_volume, meanV, N);
  k_norm_std<<<blocks, threads>>>(d_price, N, 1.0 / stdP);
  k_norm_std<<<blocks, threads>>>(d_volume, N, 1.0 / stdV);

  // Blend channels.
  k_blend<<<blocks, threads>>>(d_price, d_volume, d_pivots, d_blend, N, modeFlags,
                               priceWeight, volumeWeight, pivotWeight);
  // Single time-domain detrend using Blackman window
  k_blackman_detrend<<<blocks, threads>>>(d_blend, d_blend, N, window);
  cudaMemcpy(tmp.data(), d_blend, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("blend_after_blackman", tmp.data(), N, false);

  // Optional Hann window.
  if (modeFlags & RW_MODE_USE_HANN) {
    k_hann<<<blocks, threads>>>(d_blend, N);
    cudaMemcpy(tmp.data(), d_blend, realBytes, cudaMemcpyDeviceToHost);
    rw_log_array_sample("blend_after_hann", tmp.data(), N, false);
  }

  // FFT
  if (!checkCufft(cufftPlan1d(&plan, N, CUFFT_D2Z, 1))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  if (!checkCufft(cufftExecD2Z(plan, d_blend, d_freq))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }

  // Band-pass: remove DC and optionally tighten high freq.
  int nComplex = N / 2 + 1;
  int lowBin = (modeFlags & RW_MODE_HIGHPASS) ? std::max(1, N / window) : 0; // optional freq HP
  int highBin = (modeFlags & RW_MODE_EXTENDED_BAND) ? nComplex - 2 : std::max(4, N / 5);
  if (highBin <= lowBin) highBin = std::min(nComplex - 1, lowBin + 2);
  int bpBlocks = (nComplex + threads - 1) / threads;
  k_bandpass<<<bpBlocks, threads>>>(d_freq, nComplex, lowBin, highBin);
  rw_log("[gpu_reversal_wave_process] FFT bins: lowBin=%d highBin=%d nComplex=%d", lowBin, highBin, nComplex);

  // IFFT
  if (!checkCufft(cufftExecZ2D(plan, d_freq, d_wave))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  cufftDestroy(plan);
  plan = 0;

  // Normalize IFFT
  k_normalize<<<blocks, threads>>>(d_wave, N, 1.0 / static_cast<double>(N));

  // Copy back
  std::vector<double> h_wave(N);
  if (!checkCuda(cudaMemcpy(h_wave.data(), d_wave, realBytes, cudaMemcpyDeviceToHost))) { cleanup(); return RW_RESULT_INVALID_ARGUMENT; }
  rw_log_array_sample("wave_ifft_chrono", h_wave.data(), N, false);

  // Compute max amplitude for confidence/flags
  double maxAbs = 0.0;
  double sumSq = 0.0;
  for (double v : h_wave) { maxAbs = std::max(maxAbs, std::abs(v)); sumSq += v * v; }
  double rms = std::sqrt(sumSq / static_cast<double>(N));
  if (maxAbs < 1e-12) maxAbs = 1e-12;
  if (rms    < 1e-12) rms    = 1e-12;

  const double hiThresh = 0.6 * maxAbs;

  for (int i = 0; i < N; ++i) {
    double w = h_wave[i];
    double conf = std::min(1.0, std::abs(w) / rms);
    bool warmup = (i < window);
    int flag = 0;
    if (warmup) flag |= RW_FLAG_WARMUP;
    if (conf < 0.35) flag |= RW_FLAG_LOW_CONFIDENCE;
    if (!warmup) {
      if (w > hiThresh) flag |= RW_FLAG_BEARISH;
      else if (w < -hiThresh) flag |= RW_FLAG_BULLISH;
    }
    // write back in MT5 order (latest bar index 0)
    int mt5_idx = N - 1 - i;
    outWave[mt5_idx] = w;
    outConfidence[mt5_idx] = conf;
    outFlags[mt5_idx] = flag;
  }

  cleanup();
  rw_log("[gpu_reversal_wave_process] DONE rc=%d", RW_RESULT_OK);
  return RW_RESULT_OK;
}

// Synthetic helper: generate sinusoid + noise to validate pipeline.
__declspec(dllexport) int __stdcall gpu_reversal_wave_synthetic_test(
    int length, double oscillation, double noiseLevel,
    double* outWave, double* outConfidence, int* outFlags) {
  if (!outWave || !outConfidence || !outFlags || length <= 0)
    return RW_RESULT_INVALID_ARGUMENT;

  const double freq = (oscillation != 0.0) ? oscillation : 20.0;
  const double noise = (noiseLevel != 0.0) ? noiseLevel : 0.02;
  rw_log("[gpu_reversal_wave_synthetic_test] length=%d freq=%.2f noise=%.4f", length, freq, noise);

  double maxAbs = 0.0;
  for (int i = 0; i < length; ++i) {
    double phase = 2.0 * M_PI * i / freq;
    double r = std::fmod(1103515245u * (uint32_t)(i + 1) + 12345u, 0xFFFFFFFFu) / double(0xFFFFFFFFu);
    double w = sin(phase) + (r - 0.5) * 2.0 * noise;
    outWave[length - 1 - i] = w;  // MT5 order
    maxAbs = std::max(maxAbs, std::abs(w));
  }
  if (maxAbs < 1e-12) maxAbs = 1e-12;
  for (int i = 0; i < length; ++i) {
    double w = outWave[i];
    double conf = std::min(1.0, std::abs(w) / maxAbs);
    int flag = 0;
    if (i < (int)freq) flag |= RW_FLAG_WARMUP;
    if (conf < 0.35) flag |= RW_FLAG_LOW_CONFIDENCE;
    if (!(flag & RW_FLAG_WARMUP)) {
      if (w > 0.6 * maxAbs) flag |= RW_FLAG_BEARISH;
      else if (w < -0.6 * maxAbs) flag |= RW_FLAG_BULLISH;
    }
    outConfidence[i] = conf;
    outFlags[i] = flag;
  }
  rw_log("[gpu_reversal_wave_synthetic_test] DONE maxAbs=%.6f", maxAbs);
  return RW_RESULT_OK;
}

} // extern "C"


// ----------------------- V2: multi-wave pipeline -----------------------

static int process_wave(double* d_signal, int n, int window, int modeFlags,
                        cufftHandle plan, double* d_tmp, cufftDoubleComplex* d_freq,
                        int lowCutBins, int highCutBins,
                        std::vector<double>& h_out)
{
  const int threads = 256;
  int blocks = (n + threads - 1) / threads;

  // Single time-domain detrend (Blackman moving mean)
  k_blackman_detrend<<<blocks, threads>>>(d_signal, d_signal, n, window);

  if (modeFlags & RW_MODE_USE_HANN)
    k_hann<<<blocks, threads>>>(d_signal, n);

  if (!checkCufft(cufftExecD2Z(plan, d_signal, d_freq)))
    return RW_RESULT_INVALID_ARGUMENT;

  int nComplex = n / 2 + 1;
  int lowBin = (modeFlags & RW_MODE_HIGHPASS) ? std::max(1, lowCutBins) : 0;
  int highBin = (highCutBins > 0) ? highCutBins : nComplex - 1;
  int bpBlocks = (nComplex + threads - 1) / threads;
  k_bandpass<<<bpBlocks, threads>>>(d_freq, nComplex, lowBin, highBin);
  rw_log("[process_wave] n=%d lowBin=%d highBin=%d mode=%d", n, lowBin, highBin, modeFlags);

  if (!checkCufft(cufftExecZ2D(plan, d_freq, d_signal)))
    return RW_RESULT_INVALID_ARGUMENT;

  k_normalize<<<blocks, threads>>>(d_signal, n, 1.0 / static_cast<double>(n));

  h_out.resize(n);
  cudaMemcpy(h_out.data(), d_signal, sizeof(double) * n, cudaMemcpyDeviceToHost);
  return RW_RESULT_OK;
}

extern "C" {

__declspec(dllexport) int __stdcall gpu_reversal_wave_process_v2(
    const double* open,
    const double* price,  // close
    const double* high,
    const double* low,
    const double* volume,
    const double* pivots,
    int length,
    int window,
    int modeFlags,
    double priceWeight,
    double volumeWeight,
    double pivotWeight,
    double candleWeight,
    double* outPrice,
    double* outCandle,
    double* outZigZag,
    double* outVolume,
    double* outCombined,
    double* outConfidence,
    int* outFlags)
{
  if (!open || !price || !high || !low || !volume || !pivots ||
      !outPrice || !outCandle || !outZigZag || !outVolume || !outCombined ||
      !outConfidence || !outFlags || length <= 0 || window <= 0)
    return RW_RESULT_INVALID_ARGUMENT;
  if (length < window)
    return RW_RESULT_NOT_ENOUGH_DATA;

  const int N = length;
  const int threads = 256;
  int blocks = (N + threads - 1) / threads;

  size_t realBytes = sizeof(double) * N;
  size_t freqBytes = sizeof(cufftDoubleComplex) * (N / 2 + 1);

  // Device buffers
  double *d_open = nullptr, *d_price = nullptr, *d_high = nullptr, *d_low = nullptr, *d_volume = nullptr, *d_pivots = nullptr;
  double *d_work = nullptr;
  cufftDoubleComplex* d_freq = nullptr;
  cufftHandle plan = 0;
  bool ok = true;

  ok = ok && checkCuda(cudaMalloc(&d_open, realBytes));
  ok = ok && checkCuda(cudaMalloc(&d_price, realBytes));
  ok = ok && checkCuda(cudaMalloc(&d_high, realBytes));
  ok = ok && checkCuda(cudaMalloc(&d_low, realBytes));
  ok = ok && checkCuda(cudaMalloc(&d_volume, realBytes));
  ok = ok && checkCuda(cudaMalloc(&d_pivots, realBytes));
  ok = ok && checkCuda(cudaMalloc(&d_work, realBytes));
  ok = ok && checkCuda(cudaMalloc(&d_freq, freqBytes));
  if (!ok) {
    if (d_open) cudaFree(d_open);
    if (d_price) cudaFree(d_price);
    if (d_high) cudaFree(d_high);
    if (d_low) cudaFree(d_low);
    if (d_volume) cudaFree(d_volume);
    if (d_pivots) cudaFree(d_pivots);
    if (d_work) cudaFree(d_work);
    if (d_freq) cudaFree(d_freq);
    return RW_RESULT_INVALID_ARGUMENT;
  }

  // Copy & reverse to chronological
  k_reverse<<<blocks, threads>>>(open, d_open, N);
  k_reverse<<<blocks, threads>>>(price, d_price, N);
  k_reverse<<<blocks, threads>>>(high, d_high, N);
  k_reverse<<<blocks, threads>>>(low, d_low, N);
  k_reverse<<<blocks, threads>>>(volume, d_volume, N);
  k_reverse<<<blocks, threads>>>(pivots, d_pivots, N);

  // Sanitize inputs
  k_sanitize<<<blocks, threads>>>(d_open, N);
  k_sanitize<<<blocks, threads>>>(d_price, N);
  k_sanitize<<<blocks, threads>>>(d_high, N);
  k_sanitize<<<blocks, threads>>>(d_low, N);
  k_sanitize<<<blocks, threads>>>(d_volume, N);
  k_sanitize<<<blocks, threads>>>(d_pivots, N);

  rw_log("[gpu_reversal_wave_process_v2] length=%d window=%d mode=%d weights p/v/z/c=%.3f/%.3f/%.3f/%.3f", N, window, modeFlags, priceWeight, volumeWeight, pivotWeight, candleWeight);
  std::vector<double> tmp(N);
  cudaMemcpy(tmp.data(), d_price, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("price_chrono_after_reverse", tmp.data(), N, false);
  cudaMemcpy(tmp.data(), d_volume, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("volume_chrono_after_reverse", tmp.data(), N, false);
  cudaMemcpy(tmp.data(), d_pivots, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("pivots_chrono_after_reverse", tmp.data(), N, false);

  // Build signals on GPU
  const int volWin = 20;
  const double volScale = 2.5;
  k_candle123<<<blocks, threads>>>(d_open, d_price, d_high, d_low, d_work, N);
  // reuse d_pivots as zz step-hold output
  k_step_hold<<<1,1>>>(d_pivots, d_pivots, N);
  k_vol_zscore<<<blocks, threads>>>(d_volume, d_volume, N, volWin, volScale);
  cudaMemcpy(tmp.data(), d_work, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("candle123_chrono", tmp.data(), N, false);
  cudaMemcpy(tmp.data(), d_pivots, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("zigzag_step_chrono", tmp.data(), N, false);
  cudaMemcpy(tmp.data(), d_volume, realBytes, cudaMemcpyDeviceToHost);
  rw_log_array_sample("volume_zscore_chrono", tmp.data(), N, false);

  if (!checkCufft(cufftPlan1d(&plan, N, CUFFT_D2Z, 1))) {
    cudaFree(d_open); cudaFree(d_price); cudaFree(d_high); cudaFree(d_low); cudaFree(d_volume); cudaFree(d_pivots);
    cudaFree(d_work); cudaFree(d_freq);
    return RW_RESULT_INVALID_ARGUMENT;
  }

  // Process each wave
  std::vector<double> h_wave_price, h_wave_candle, h_wave_zz, h_wave_vol;
  int lowCut = (modeFlags & RW_MODE_HIGHPASS) ? std::max(1, N / window) : 0; // optional freq detrend
  int highCut = (modeFlags & RW_MODE_EXTENDED_BAND) ? N/2 : std::max(4, N/5);
  if (highCut > N/2) highCut = N/2;
  if (highCut <= lowCut) highCut = std::min(N/2, lowCut + 2);

  if (process_wave(d_price, N, window, modeFlags, plan, d_work, d_freq, lowCut, highCut, h_wave_price) != RW_RESULT_OK ||
      process_wave(d_work, N, window, modeFlags, plan, d_price, d_freq, lowCut, highCut, h_wave_candle) != RW_RESULT_OK ||
      process_wave(d_pivots, N, window, modeFlags, plan, d_price, d_freq, lowCut, highCut, h_wave_zz) != RW_RESULT_OK ||
      process_wave(d_volume, N, window, modeFlags, plan, d_price, d_freq, lowCut, highCut, h_wave_vol) != RW_RESULT_OK) {
    cufftDestroy(plan);
    cudaFree(d_open); cudaFree(d_price); cudaFree(d_high); cudaFree(d_low); cudaFree(d_volume); cudaFree(d_pivots);
    cudaFree(d_work); cudaFree(d_freq);
    return RW_RESULT_INVALID_ARGUMENT;
  }

  cufftDestroy(plan);

  // Combined and flags/confidence
  std::vector<double> h_comb(N);
  double sumW = std::abs(priceWeight) + std::abs(volumeWeight) + std::abs(pivotWeight) + std::abs(candleWeight);
  if (sumW < 1e-12) sumW = 1.0;
  for (int i = 0; i < N; ++i) {
    h_comb[i] = (priceWeight * h_wave_price[i] +
                 volumeWeight * h_wave_vol[i] +
                 pivotWeight * h_wave_zz[i] +
                 candleWeight * h_wave_candle[i]) / sumW;
  }

  rw_log_array_sample("wave_price_chrono", h_wave_price.data(), N, false);
  rw_log_array_sample("wave_candle_chrono", h_wave_candle.data(), N, false);
  rw_log_array_sample("wave_zigzag_chrono", h_wave_zz.data(), N, false);
  rw_log_array_sample("wave_volume_chrono", h_wave_vol.data(), N, false);
  rw_log_array_sample("wave_combined_chrono", h_comb.data(), N, false);

  double maxAbs = 0.0, sumSq = 0.0;
  for (double v : h_comb) { maxAbs = std::max(maxAbs, std::abs(v)); sumSq += v * v; }
  double rms = sqrt(sumSq / static_cast<double>(N));
  if (maxAbs < 1e-12) maxAbs = 1e-12;
  if (rms < 1e-12) rms = 1e-12;
  double hiThresh = 0.6 * maxAbs;

  for (int i = 0; i < N; ++i) {
    bool warmup = (i < window);
    double conf = std::min(1.0, std::abs(h_comb[i]) / rms);
    int flag = 0;
    if (warmup) flag |= RW_FLAG_WARMUP;
    if (conf < 0.35) flag |= RW_FLAG_LOW_CONFIDENCE;
    if (!warmup) {
      if (h_comb[i] > hiThresh) flag |= RW_FLAG_BEARISH;
      else if (h_comb[i] < -hiThresh) flag |= RW_FLAG_BULLISH;
    }
    int mt5_idx = N - 1 - i;
    outPrice[mt5_idx] = h_wave_price[i];
    outCandle[mt5_idx] = h_wave_candle[i];
    outZigZag[mt5_idx] = h_wave_zz[i];
    outVolume[mt5_idx] = h_wave_vol[i];
    outCombined[mt5_idx] = h_comb[i];
    outConfidence[mt5_idx] = conf;
    outFlags[mt5_idx] = flag;
  }

  rw_log_array_sample("outPrice_mt5", outPrice, N, true);
  rw_log_array_sample("outCandle_mt5", outCandle, N, true);
  rw_log_array_sample("outZigZag_mt5", outZigZag, N, true);
  rw_log_array_sample("outVolume_mt5", outVolume, N, true);
  rw_log_array_sample("outCombined_mt5", outCombined, N, true);
  rw_log_array_sample("outConfidence_mt5", outConfidence, N, true);
  rw_log("[gpu_reversal_wave_process_v2] DONE rc=%d", RW_RESULT_OK);

  cudaFree(d_open); cudaFree(d_price); cudaFree(d_high); cudaFree(d_low); cudaFree(d_volume); cudaFree(d_pivots);
  cudaFree(d_work); cudaFree(d_freq);
  return RW_RESULT_OK;
}

} // extern "C"
