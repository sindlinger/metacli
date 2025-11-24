$ErrorActionPreference = 'Stop'

param(
  [string]$TerminalDir,
  [string]$TemplateSrc = "$(Join-Path (Get-Location) 'factory/templates/mtcli-default.tpl')"
)

if (-not (Test-Path $TerminalDir)) { throw "TerminalDir não existe: $TerminalDir" }
if (-not (Test-Path $TemplateSrc)) { throw "Template não encontrado: $TemplateSrc" }

$tplDest = Join-Path $TerminalDir 'MQL5/Profiles/Templates/mtcli-default.tpl'
New-Item -ItemType Directory -Force -Path (Split-Path $tplDest) | Out-Null
Copy-Item $TemplateSrc $tplDest -Force

# Salva como Default.tpl também, para o MT5 carregar no perfil padrão
$defaultTpl = Join-Path $TerminalDir 'MQL5/Profiles/Templates/Default.tpl'
Copy-Item $TemplateSrc $defaultTpl -Force

Write-Host "Template aplicado em $tplDest e $defaultTpl"
