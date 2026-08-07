# Self-host on Windows LAN — requires UAII_DASH_TOKEN
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not $env:UAII_DASH_TOKEN) {
  Write-Error "Set UAII_DASH_TOKEN first, e.g. `$env:UAII_DASH_TOKEN = -join ((1..32) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) })"
}
if (-not (Test-Path "node_modules")) { npm run install:all }
npm run build
$env:NODE_ENV = "production"
$env:UAII_DASH_BIND = if ($env:UAII_DASH_BIND) { $env:UAII_DASH_BIND } else { "0.0.0.0" }
if (-not $env:UAII_DASH_PORT) { $env:UAII_DASH_PORT = "8787" }
Write-Host "Self-host http://$($env:UAII_DASH_BIND):$($env:UAII_DASH_PORT)  (Bearer token required)"
npm start
