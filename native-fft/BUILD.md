# fasttransforms.dll (CUDA 13.0) — build rápido

Requisitos
----------
- CUDA Toolkit 13.0 instalado (nvcc no PATH)
- MSVC 64-bit (cl.exe) ou toolset Visual Studio 2022
- GPU alvo: RTX A4500 (compute 8.6). Ajuste `CUDA_ARCHITECTURES` se usar outra GPU.
- CMake 3.23+

Passo a passo (PowerShell ou Developer Command Prompt)
------------------------------------------------------
```powershell
cd native-fft
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

O binário resultante ficará em `native-fft/build/fasttransforms.dll`.
Copie-o para:
```
C:\Users\pichau\AppData\Roaming\MetaQuotes\Terminal\72D7079820AB4E374CDC07CD933C3265\MQL5\Libraries\fasttransforms.dll
```

Observações
-----------
- APIs expostas (`__stdcall`):
  - `gpu_reversal_wave_process`
  - `gpu_reversal_wave_synthetic_test`
  - `gpu_reversal_wave_process_v2` (usa open, close, high, low, volume, pivots; entrega 4 waves + combinada)
- Nenhum arquivo MQL é modificado; só o DLL precisa ser colocado na pasta de Libraries do MT5.
- Se precisar alterar a GPU alvo, edite `CUDA_ARCHITECTURES` em `CMakeLists.txt`.
