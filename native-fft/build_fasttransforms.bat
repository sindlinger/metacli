@echo off
setlocal

REM Caminho para o cmake 3.22+ (ajuste se precisar)
set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"

if not exist "%CMAKE_EXE%" (
  echo Nao encontrei %CMAKE_EXE%. Ajuste a variavel CMAKE_EXE no arquivo .bat.
  exit /b 1
)

cd /d %~dp0

"%CMAKE_EXE%" -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release || exit /b 1
"%CMAKE_EXE%" --build build --config Release || exit /b 1

echo.
echo Copie o DLL gerado:
echo   %~dp0build\fasttransforms.dll
echo para:
echo   C:\Users\pichau\AppData\Roaming\MetaQuotes\Terminal\72D7079820AB4E374CDC07CD933C3265\MQL5\Libraries\
echo.

endlocal
