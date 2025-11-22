# Reversal Wave GPU Pipeline

This document tracks the current state of the reversal wave pipeline that powers
`mql5/Indicators/ReversalWave.mq5`. The goal is to keep all heavy processing
inside the native DLL so the indicator remains a thin shell.

## Input Signals

The DLL expects three synchronous series (newest bar at index 0):

- **Price** – the indicator passes the `close[]` series. Inside the DLL the
  series is converted to chronological order, detrended with a linear fit and
  normalized to zero mean / unit variance.
- **Volume** – the raw `volume[]` series becomes a z-score channel so that
  unusual activity is highlighted without affecting amplitude drastically.
- **Pivots** – the indicator uses a ZigZag handle to emit `+1` for local tops
  and `-1` for local bottoms; everything else is `0`. The DLL smooths this
  channel with a lightweight EMA and normalizes it.

All buffers are contiguous so the GPU version can move them to device memory in
single chunks later.

## Pipeline Stages inside the DLL

1. **Series preparation** – convert MT5 series (index 0 = latest bar) into
   chronological vectors so the FFT operates on a natural timeline and sanitize
   any `NaN`/`Inf` values back to zero.
2. **Detrend & normalization** – remove a linear trend, normalize price and
   volume channels, and smooth the pivot signal. This keeps the FFT energy in a
   predictable range.
3. **Channel blending** – combine the channels with configurable weights. A
   `mode` bit flag can boost the pivot channel when a user wants structural
   signals to dominate.
4. **Windowing** – optional Hann windowing (also controlled by mode flags)
   minimizes spectral leakage before the FFT.
5. **FFT + band filtering** – ALGLIB’s `fftr1d` provides the forward transform.
   A configurable band-pass removes DC components (high-pass) and clamps noisy
   high frequencies. Flags such as `RW_MODE_EXTENDED_BAND` widen the band.
6. **IFFT & scaling** – the inverse transform rebuilds a smoothed wave in the
   time domain. The DLL normalizes the amplitude and computes an energy-based
   confidence per bar.
7. **Flag extraction** – the code looks for local extrema with sufficient
   normalized amplitude and confidence to label tentative top/bottom flags,
   skipping warmup bars where metrics have not stabilized yet.
8. **Series order restoration** – wave, confidence, and flag buffers are written
   back in MT5 order (index 0 = latest bar) so the indicator can copy them
   straight into its buffers.

## DLL API

The DLL exports the following entry points (declared in
`native-fft/include/fasttransforms.h`):

```c
int __stdcall gpu_reversal_wave_process(
    const double* price,
    const double* volume,
    const double* pivots,
    int length,
    int window,
    int modeFlags,
    double priceWeight,
    double volumeWeight,
    double pivotWeight,
    double* outWave,
    double* outConfidence,
    int* outFlags);

int __stdcall gpu_reversal_wave_synthetic_test(
    int length,
    double oscillation,
    double noiseLevel,
    double* outWave,
    double* outConfidence,
    int* outFlags);
```

Mode flags:

| Flag | Value | Effect |
| ---- | ----- | ------ |
| `RW_MODE_HIGHPASS` | 1 | Removes the slow trend before FFT, useful for highlighting swings. |
| `RW_MODE_EMPHASIZE_PIVOT` | 2 | 50% gain bump on the pivot channel before blending. |
| `RW_MODE_USE_HANN` | 4 | Applies a Hann window prior to the FFT. |
| `RW_MODE_EXTENDED_BAND` | 8 | Slightly widens the high-frequency guard band.

Output flags (`outFlags` bitmask):

| Flag | Value | Meaning |
| ---- | ----- | ------- |
| `RW_FLAG_BULLISH` | 1 | Local trough with enough confidence (candidate bullish reversal). |
| `RW_FLAG_BEARISH` | 2 | Local peak with enough confidence (candidate bearish reversal). |
| `RW_FLAG_LOW_CONFIDENCE` | 4 | Energy/amplitude too small; treat the bar with caution. |
| `RW_FLAG_WARMUP` | 8 | Early bars of the buffer still warming up the sliding metrics. |

Return codes are defined in the same header (`RW_RESULT_*`). The exported
synthetic helper generates a sinusoidal test case and routes through the same
pipeline, making it easy to sanity-check the DLL without MT5.

## MQL5 Integration

- `mql5/Include/ReversalWaveBridge.mqh` imports the DLL exports and adds a small
  helper `RWComputeWave()` that logs descriptive errors.
- `mql5/Indicators/ReversalWave.mq5` gathers the close, volume, and ZigZag pivot
  signals, calls `RWComputeWave`, and copies the resulting wave, confidence and
  flags into indicator buffers. Bullish/bearish plotting uses the respective flag
  bits; low-confidence/warmup bits stay available for styling but do not trigger
  arrows. The indicator itself performs no heavy math.
- The ZigZag parameters are exposed as input fields so the user can keep the
  pivot signal aligned with their setup.

## Tests and Validation

- `native-fft` exposes `gpu_reversal_wave_synthetic_test` which can be called
  from any harness to ensure the pipeline responds to controlled oscillations.
- `mql5/Scripts/ReversalWaveSelfTest.mq5` uses that exported helper, prints the
  average confidence plus counts for bullish, bearish, low-confidence, and
  warmup bars, and acts as a quick smoke test.
- The indicator can be attached to live charts or the Strategy Tester; when the
  DLL returns `RW_RESULT_NOT_ENOUGH_DATA` the indicator waits for more history
  instead of performing partial calculations.

## Known Limitations / Next Steps

- The current ZigZag-based pivot input is simplistic. Feeding a richer 1‑2‑3
  detector once it lives in an EA would likely improve the structural channel.
- The DLL runs on CPU today. The data layout and batching already assume a
  contiguous memory layout so porting the FFT + windowing to CUDA/OpenCL will be
  straightforward when the hardware path is ready.
- Band-pass parameters are inferred from the window size. Future work could add
  explicit frequency controls exposed through the API.
