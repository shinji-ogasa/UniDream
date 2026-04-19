param(
  [string]$CacheDir = "checkpoints\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$Symbol = "BTCUSDT",
  [string]$Interval = "15m",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01",
  [string]$Device = "auto",
  [string]$OutputDir = "checkpoints\source_family_suite_free"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[FREE-E2E] Fetching free sources..."
& powershell -ExecutionPolicy Bypass -File .\scripts\fetch_free_source_rollout.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -Symbol $Symbol `
  -Interval $Interval `
  -Start $Start `
  -End $End

Write-Host "[FREE-E2E] Running free source family suite..."
& powershell -ExecutionPolicy Bypass -File .\scripts\run_free_source_family_suite.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -Start $Start `
  -End $End `
  -Device $Device `
  -OutputDir $OutputDir

Write-Host "[FREE-E2E] Writing rollout snapshot..."
& powershell -ExecutionPolicy Bypass -File .\scripts\write_source_rollout_snapshot.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -SuiteConfig .\configs\source_rollout_suite_free.yaml `
  -OutputPath (Join-Path $OutputDir "free_source_rollout_snapshot.json")

Write-Host "[FREE-E2E] Done."
