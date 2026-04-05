param(
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$OutputManifest = "configs\\source_manifest_next_stage.yaml"
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"

Write-Host "[PLAN] Diagnose current rollout ..."
& powershell -ExecutionPolicy Bypass -File .\scripts\diagnose_source_rollout.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag

Write-Host ""
Write-Host "[PLAN] Generate next-stage manifest ..."
& powershell -ExecutionPolicy Bypass -File .\scripts\generate_next_source_manifest.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -Output $OutputManifest

Write-Host ""
Write-Host "[PLAN] Dry-run next-stage manifest ..."
& $Python build_source_cache_from_manifest.py `
  --manifest $OutputManifest `
  --dry-run
