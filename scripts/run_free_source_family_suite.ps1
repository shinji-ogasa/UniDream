param(
  [string]$CacheDir = "checkpoints\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01",
  [string]$Device = "auto",
  [string]$OutputDir = "checkpoints\source_family_suite_free"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

& powershell -ExecutionPolicy Bypass -File .\scripts\run_source_family_suite.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -Device $Device `
  -SuiteDir $OutputDir `
  -SuiteConfig configs\source_rollout_suite_free.yaml `
  -SkipMissing $true

& powershell -ExecutionPolicy Bypass -File .\scripts\select_best_source_family.ps1 `
  -SummaryCsv (Join-Path $OutputDir "suite_summary.csv") `
  -OutputPath (Join-Path $OutputDir "best_source_family.md")
