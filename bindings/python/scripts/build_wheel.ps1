# Build a UAII Python wheel with bundled native uaii_capi (Windows).
# Run from repo root or bindings/python.
#
# Prerequisites: CMake, a C++ toolchain, Python 3.10+ with pip.
#
# Usage:
#   powershell -File bindings/python/scripts/build_wheel.ps1
#   powershell -File bindings/python/scripts/build_wheel.ps1 -BuildDir build -Config Release

param(
  [string]$BuildDir = "build",
  [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RepoRoot = Resolve-Path (Join-Path $PyRoot "../..")

Push-Location $RepoRoot
try {
  Write-Host "==> cmake configure (tip: add -DUAII_BUILD_PYTHON=ON for pybind)"
  cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=$Config -DUAII_BUILD_PLUGINS=ON
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  Write-Host "==> cmake --build $BuildDir --config $Config"
  cmake --build $BuildDir --config $Config --parallel
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  Write-Host "==> bundle_native.py"
  python (Join-Path $PyRoot "scripts/bundle_native.py") --build-dir $BuildDir
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  Write-Host "==> python -m build"
  python -m pip install -q build
  python -m build $PyRoot
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  Write-Host "Wheel(s) under: $(Join-Path $PyRoot 'dist')"
} finally {
  Pop-Location
}
