param(
  [string]$CacheDir = "checkpoints\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01",
  [string]$OutJson = "checkpoints\coinmetrics_adractcnt.json"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$uri = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=AdrActCnt&frequency=1d&start_time=$Start&end_time=$End"
Write-Host "[CM] Downloading AdrActCnt JSON..."
Invoke-WebRequest -Uri $uri -OutFile $OutJson

Write-Host "[CM] Converting export to source cache..."
uv run python .\build_coinmetrics_source_cache.py `
  --cache-dir $CacheDir `
  --cache-tag $CacheTag `
  --asset btc `
  --start $Start `
  --end $End `
  --frequency 1d `
  --input-file $OutJson `
  --metric active_address_growth=AdrActCnt:logdiff
