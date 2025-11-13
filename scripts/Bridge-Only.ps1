param(
  [string]$Root = "C:\mql5\MQL5\alglib",
  [string]$Config = "Release",
  [string]$Arch = "x64",
  [int]$Parallel = 24,
  [string]$Libs = ""
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERR ] $msg" -ForegroundColor Red }

function Test-Admin {
  try { return ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) }
  catch { return $false }
}

function Find-LibrariesPath {
  param([string]$Hint)
  if ($Hint -and (Test-Path $Hint)) { return (Resolve-Path $Hint).Path }
  $base = Join-Path $env:APPDATA 'MetaQuotes\Terminal'
  if (-not (Test-Path $base)) { return $null }
  $candidates = Get-ChildItem $base -Directory | ForEach-Object {
    $libs = Join-Path $_.FullName 'MQL5\Libraries'
    if (Test-Path $libs) { [PSCustomObject]@{ Path=$libs; Stamp=(Get-Item $_.FullName).LastWriteTimeUtc } }
  } | Sort-Object Stamp -Descending
  if ($candidates -and $candidates.Count -gt 0) { return $candidates[0].Path }
  return $null
}

function Ensure-CMake {
  $cm = Get-Command cmake -ErrorAction SilentlyContinue
  if (-not $cm) { throw 'cmake not found in PATH. Open Developer PowerShell for VS or install CMake.' }
}

function Kill-Processes {
  param([string[]]$Names)
  foreach($n in $Names){ try { & taskkill /F /IM $n | Out-Null } catch { } }
}

function Copy-WithBackup {
  param([string]$Src,[string]$Dst,[string]$BackupRoot)
  New-Item -ItemType Directory -Force -Path (Split-Path $Dst) | Out-Null
  if (Test-Path $Dst) {
    $termId = Split-Path (Split-Path (Split-Path $Dst)) -Leaf
    $bdst = Join-Path (Join-Path $BackupRoot $termId) (Split-Path $Dst -Leaf)
    New-Item -ItemType Directory -Force -Path (Split-Path $bdst) | Out-Null
    Copy-Item $Dst $bdst -Force
    Write-Info "BACKUP  $Dst -> $bdst"
  }
  Copy-Item $Src $Dst -Force
  Write-Info "COPY    $Src -> $Dst"
}

Write-Host "=== Bridge-Only rebuild+deploy ===" -ForegroundColor Green
if (-not (Test-Admin)) { Write-Warn 'Run as Administrator for best results (process kill/copy). Proceeding anyway.' }

Ensure-CMake
$build = Join-Path $Root 'build_bridge'
if (Test-Path $build) { Write-Info "Cleaning $build"; Remove-Item -Recurse -Force $build }

Write-Info "Configuring CMake ($Arch/$Config)"
& cmake -S $Root -B $build -A $Arch

Write-Info "Building alglib_bridge ($Config, -j $Parallel)"
& cmake --build $build --config $Config --target alglib_bridge -j $Parallel

$dll = Join-Path (Join-Path $build $Config) 'alglib_bridge.dll'
if (-not (Test-Path $dll)) { throw "Build succeeded but artifact not found: $dll" }

if (-not $Libs) { $Libs = Find-LibrariesPath -Hint $Libs }
if (-not $Libs) { throw 'Could not locate MQL5\Libraries automatically. Use -Libs C:\\... to specify.' }
$dst = Join-Path $Libs 'alglib_bridge.dll'

Write-Info "Deploying to $Libs"
$backupRoot = Join-Path $Root 'tools\mtcli_backups'
Copy-WithBackup -Src $dll -Dst $dst -BackupRoot $backupRoot

Write-Host "Done." -ForegroundColor Green

