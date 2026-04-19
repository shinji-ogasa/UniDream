param(
  [string]$CacheDir = "checkpoints\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$ZipDir = "raw\binance_klines",
  [string]$CoinMetricsJson = "raw\coinmetrics\btc_adractcnt.json",
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

Write-Host "[FREE-MANUAL] Importing Binance public spot zips..."
uv run python .\build_binance_public_spot_series.py `
  --cache-dir $CacheDir `
  --cache-tag $CacheTag `
  --symbol $Symbol `
  --interval $Interval `
  --start $Start `
  --end $End `
  --zip-dir $ZipDir `
  --write-buy-sell-ratio

Write-Host "[FREE-MANUAL] Importing Coin Metrics export..."
uv run python .\build_coinmetrics_source_cache.py `
  --cache-dir $CacheDir `
  --cache-tag $CacheTag `
  --asset btc `
  --start $Start `
  --end $End `
  --frequency 1d `
  --input-file $CoinMetricsJson `
  --metric active_address_growth=AdrActCnt:logdiff

Write-Host "[FREE-MANUAL] Running free source family suite..."
& powershell -ExecutionPolicy Bypass -File .\scripts\run_free_source_family_suite.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -Start $Start `
  -End $End `
  -Device $Device `
  -OutputDir $OutputDir
