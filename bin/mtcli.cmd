@echo off
setlocal
for %%I in ("%~dp0\..") do set "ROOT_DIR=%%~fI"
set "DIST=%ROOT_DIR%\dist\cli.js"
if not exist "%DIST%" (
  echo [mtcli] dist\cli.js nao encontrado. Execute "npm install" e "npm run build" primeiro. 1>&2
  exit /b 2
)
node "%DIST%" %*
exit /b %ERRORLEVEL%
