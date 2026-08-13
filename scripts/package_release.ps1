# Package an unsigned UAII release zip (Windows-oriented helper).
# Does NOT code-sign. See docs/signing.md for org cert / signtool / notarize.
#
# Usage (from repo root):
#   powershell -File scripts/package_release.ps1
#   powershell -File scripts/package_release.ps1 -BuildDir build -OutDir dist

param(
  [string]$BuildDir = "build",
  [string]$OutDir = "dist",
  [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")
Set-Location $RepoRoot

$candidates = @(
  (Join-Path $BuildDir "libs/uaii-cli/uaii.exe"),
  (Join-Path $BuildDir "libs/uaii-cli/$Config/uaii.exe"),
  (Join-Path $BuildDir "libs/uaii-cli/uaii"),
  (Join-Path $BuildDir "libs/uaii-cli/$Config/uaii"),
  (Join-Path $BuildDir "uaii.exe"),
  (Join-Path $BuildDir "uaii")
)

$bin = $null
foreach ($c in $candidates) {
  if (Test-Path -LiteralPath $c) {
    $bin = Resolve-Path $c
    break
  }
}

if (-not $bin) {
  Write-Error "uaii binary not found under '$BuildDir'. Build first (cmake --build $BuildDir --target uaii)."
}

$stage = Join-Path $OutDir "_uaii_release_stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$destName = if ($bin.Path -match '\.exe$') { "uaii.exe" } else { "uaii" }
Copy-Item -LiteralPath $bin.Path -Destination (Join-Path $stage $destName)
Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination (Join-Path $stage "LICENSE") -ErrorAction SilentlyContinue
if (-not (Test-Path (Join-Path $stage "LICENSE"))) {
  Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE.md") -Destination (Join-Path $stage "LICENSE") -ErrorAction SilentlyContinue
}
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination (Join-Path $stage "README.md")

$version = "0.2.0"
$zipName = "uaii-$version-unsigned-win-x64.zip"
if ($IsLinux -or $env:OS -notmatch "Windows") {
  $zipName = "uaii-$version-unsigned.zip"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$zipPath = Join-Path $OutDir $zipName
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force
Remove-Item -Recurse -Force $stage

Write-Host "Created unsigned package: $zipPath"
Write-Host "NOT signed — see docs/signing.md for Authenticode / notarization with org certificates."
