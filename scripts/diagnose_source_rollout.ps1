param(
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2"
)

$ErrorActionPreference = "Stop"

Write-Host "[DIAG] Stage status"
& .\scripts\report_source_stage_status.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag

Write-Host ""
Write-Host "[DIAG] Next source step"
& .\scripts\recommend_next_source_step.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag

Write-Host ""
Write-Host "[DIAG] Fetch command hints"
& .\scripts\recommend_source_fetch_commands.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag
