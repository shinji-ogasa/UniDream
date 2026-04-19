param(
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01",
  [string]$Device = "auto",
  [string]$OutputDir = "checkpoints\\source_family_suite_free"
)

$ErrorActionPreference = "Stop"

Write-Host "[Issue6] free source rollout"
& powershell -ExecutionPolicy Bypass -File .\scripts\run_free_source_rollout_end_to_end.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -Start $Start `
  -End $End `
  -Device $Device `
  -OutputDir $OutputDir

Write-Host "[Issue6] best source family"
uv run python select_best_source_family.py `
  --summary (Join-Path $OutputDir "suite_summary.csv") `
  --output (Join-Path $OutputDir "best_source_family.md")
