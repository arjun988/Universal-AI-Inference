# Local dashboard on Windows (loopback)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path "node_modules")) { npm run install:all }
npm run build
$env:NODE_ENV = "production"
if (-not $env:UAII_DASH_BIND) { $env:UAII_DASH_BIND = "127.0.0.1" }
if (-not $env:UAII_DASH_PORT) { $env:UAII_DASH_PORT = "8787" }
Write-Host "UAII Dashboard  http://$($env:UAII_DASH_BIND):$($env:UAII_DASH_PORT)"
npm start
