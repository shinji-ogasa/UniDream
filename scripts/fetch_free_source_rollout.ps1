param(
  [string]$CacheDir = "checkpoints\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$Symbol = "BTCUSDT",
  [string]$Interval = "15m",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[FREE] Fetching Binance basis + spot orderflow proxy..."
& .\.venv\Scripts\python.exe .\build_binance_source_cache.py `
  --cache-dir $CacheDir `
  --cache-tag $CacheTag `
  --symbol $Symbol `
  --interval $Interval `
  --start $Start `
  --end $End `
  --skip-oi `
  --prefer-spot-taker-proxy

Write-Host "[FREE] Fetching Coin Metrics active addresses..."
try {
  & .\.venv\Scripts\python.exe .\build_coinmetrics_source_cache.py `
    --cache-dir $CacheDir `
    --cache-tag $CacheTag `
    --asset btc `
    --start $Start `
    --end $End `
    --frequency 1d `
    --metric active_address_growth=AdrActCnt:logdiff
}
catch {
  Write-Host "[FREE] Python Coin Metrics fetch failed. Falling back to PowerShell download + local import..."
  & powershell -ExecutionPolicy Bypass -File .\scripts\fetch_coinmetrics_active_addresses.ps1 `
    -CacheDir $CacheDir `
    -CacheTag $CacheTag `
    -Start $Start `
    -End $End
}

Write-Host "[FREE] Done. Current rollout status:"
& powershell -ExecutionPolicy Bypass -File .\scripts\report_source_stage_status.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -SuiteConfig configs\source_rollout_suite_free.yaml
